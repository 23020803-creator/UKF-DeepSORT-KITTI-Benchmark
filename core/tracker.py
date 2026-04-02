# Class quản lý tổng (Ghép nối YOLO, Kalman và Matching).
import numpy as np
import logging
from core.track import Tracker, TrackState
from core.ukf import UKF
from core import matching

logger = logging.getLogger(__name__)

class Tracker:
    """
    Trái tim của hệ thống MOT (Multi-Object Tracking).
    Điều phối UKF (Động học), OSNet (Ngoại hình) và Thuật toán Cascade Matching.
    """
    def __init__(self, max_age=30, n_init=3, cosine_threshold=0.25, max_tracks=150):
        self.max_age = max_age
        self.n_init = n_init
        self.cosine_threshold = cosine_threshold
        self.max_tracks = max_tracks
        self.tracks = []
        self.next_id = 1
        self.ukf_engine = UKF()
    
    def predict(self):
        """
        Dự đoán mù cho tất cả các đối tượng.
        """
        for track in self.tracks:
            track.predict(self.ukf_engine)
    
    def __get_safe_feature(self, det):
        """
        Đảm bảo Feature luôn hợp lệ vfa chuẩn hóa L2.
        """
        feat = det.get('feature')
        if feat is None or not np.instance(feat).all():
            feat = np.random.normal(0, 1e-5,512).astype(np.float32)

        feat /= (np.linalg.norm(feat) + 1e-6)
        return feat
    
    def update(self, detections, frame_id=0):
        """
        Cập nhật dữ liệu quan sát thực tế và quản lý vòng đời của các track.
        """
        valid_detections = [d for d in detections if d.get('conf', 0) > 0.3]

        # Thuật toán ghép cặp phân cấp (Cascade Matching)
        matches, unmatched_tracks, unmatched_detections = matching.cascade_matching(self.tracks, valid_detections, self.ukf_engine, self.max_age, self.cosine_threshold)

        # Cập nhật các track đã ghép cặp
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = valid_detections[det_idx]
            safe_feat = self._get_safe_feature(det)
            track.update(self.ukf_engine, det['bbox'], safe__feat)

        # Xử lý đối tượng mất dấu.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Khởi tạo track mới cho các phát hiện chưa ghép cặp.
        for det_idx in unmatched_detections:
            det = valid_detections[det_idx]

            if det.get('conf', 0) > 0.5:
                safe_feat = self._get_safe_feature(det)

                if not self._is_duplicate_track(det['bbox'], safe_feat):
                    self._initiate_track(det['bbox'], safe_feat)

        # Loại bỏ các track đã chết.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        if len(self.tracks) > self.max_tracks:
            # Ưu tiên giữ: Đã xác nhận (Confirmed) -> Mới thấy (Update gần) -> Tồn tại lâu (Hits)
            self.tracks.sort(key=lambda t: (0 if t.is_confirmed() else 1, t.time_since_update, -t.hits, getattr(t, 'age', 0) * -1))
            self.tracks = self.tracks[:self.max_tracks]

        
    def _is_duplicate_track(self, det_bbox, det_feature, iou_threshold=0.3, cos_sim_threshold=0.8):
        """
        Gating 3 lớp: Hình học (IoU) + Động học (Adaptive Maha) + Ngoại hình (EMA Cosine).
        """
        if not self.tracks:
            return False

        d_area = det_bbox[2] * det_bbox[3]
        if d_area <= 0: return False
        
        # Chuyển detection bbox sang chuẩn UKF [cx, cy, a, h]
        bbox_ukf = np.array([
            det_bbox[0] + det_bbox[2] / 2.0,
            det_bbox[1] + det_bbox[3] / 2.0,
            det_bbox[2] / max(det_bbox[3], 1e-3),
            det_bbox[3]
        ], dtype=np.float32)

        for t in self.tracks:
            # Chỉ check duplicate với các track vừa mới nhìn thấy (<= 2 frames)
            if t.is_confirmed() and t.time_since_update <= 2:
                t_bbox = t.to_tlwh()
                t_area = t_bbox[2] * t_bbox[3]
                
                # Tính Intersection nhanh
                x_left = max(det_bbox[0], t_bbox[0])
                y_top = max(det_bbox[1], t_bbox[1])
                x_right = min(det_bbox[0] + det_bbox[2], t_bbox[0] + t_bbox[2])
                y_bottom = min(det_bbox[1] + det_bbox[3], t_bbox[1] + t_bbox[3])
                
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    iou = intersection / (d_area + t_area - intersection)
                    
                    if iou > iou_threshold:
                        # Nới lỏng ngưỡng theo độ trễ khung hình
                        dynamic_maha_thresh = 9.4877 + (0.5 * t.time_since_update)
                        maha_dist = self.ukf_engine.gating_distance(t.mean, t.covariance, bbox_ukf)
                        
                        if maha_dist > dynamic_maha_thresh:
                            continue 

                        # So sánh feature EMA của track với detection mới
                        t_feat = t.get_feature() # Đã được L2-normalized trong track.py
                        cos_sim = np.dot(t_feat, det_feature)
                        
                        # Nếu trùng khớp cả 3 yếu tố -> Khẳng định là Duplicate
                        if cos_sim > cos_sim_threshold:
                            return True
        return False

    def _initiate_track(self, detection, safe_feature):
        """Tạo hồ sơ theo dõi mới (Track) cho vật thể."""
        bbox = detection['bbox'] 
        bbox_ukf = np.array([
            bbox[0] + bbox[2] / 2.0,
            bbox[1] + bbox[3] / 2.0,
            bbox[2] / max(bbox[3], 1e-3),
            bbox[3]
        ], dtype=np.float32)

        mean, covariance = self.ukf_engine.initiate(bbox_ukf)
        
        new_track = Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            class_id=detection['class_id'],
            feature=safe_feature
        )
        
        new_track._max_age = self.max_age
        new_track._n_init = self.n_init
        
        self.tracks.append(new_track)
        self._next_id += 1

    def get_results(self):
        """Lấy danh sách các Track an toàn (Confirmed) để hiển thị."""
        results = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            
            results.append({
                'track_id': track.track_id,
                'class_id': track.class_id,
                'bbox': track.to_tlwh()
            })
        return results