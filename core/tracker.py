import numpy as np
import logging

# SỬA LỖI IMPORT: Gọi đúng class Track thay vì Tracker
from core.track import Track, TrackState  
from core.ukf import UKF
from core import matching

logger = logging.getLogger(__name__)

class Detection:
    """
    [THÊM MỚI] Lớp phụ trợ (Wrapper Class).
    Bản chất: Gói các mảng rời rạc (bbox, conf, class_id, feature) thành một Đối tượng (Object).
    Tại sao phải làm vậy? Vì file matching.py cần gọi `det.class_id` để kiểm tra khác loại.
    """
    def __init__(self, bbox, conf, class_id, feature):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.conf = float(conf)
        self.class_id = int(class_id)
        self.feature = np.asarray(feature, dtype=np.float32)

class Tracker:
    """
    Trái tim của hệ thống MOT (Multi-Object Tracking).
    Điều phối UKF (Động học), OSNet (Ngoại hình) và Thuật toán Matching.
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
        Bước 1: Dự đoán vị trí của tất cả các xe đang theo dõi (trước khi nhận ảnh mới).
        """
        for track in self.tracks:
            track.predict(self.ukf_engine)
    
    def update(self, bboxes, confs, class_ids, features):
        """
        Bước 2: Cập nhật dữ liệu từ YOLO & ReID vào hệ thống.
        [SỬA LỖI ĐẦU VÀO]: Nhận 4 mảng độc lập thay vì nhận list of dicts.
        """
        valid_detections = []
        
        # Lọc các bounding box có độ tin cậy thấp và bọc thành Object Detection
        if len(bboxes) > 0:
            for bbox, conf, cls_id, feat in zip(bboxes, confs, class_ids, features):
                if conf > 0.3:
                    valid_detections.append(Detection(bbox, conf, cls_id, feat))

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(valid_detections)))

        # SỬA LỖI MATCHING: Loại bỏ `cascade_matching` ảo, dùng đúng quy trình toán học
        if len(self.tracks) > 0 and len(valid_detections) > 0:
            # 1. Thu thập ma trận đặc trưng
            features_old = np.array([t.get_feature() for t in self.tracks])
            features_new = np.array([d.feature for d in valid_detections])

            # 2. Tính Cost Matrix bằng Cosine (Gọi sang matching.py)
            cost_matrix = matching.compute_cosine_distance(features_old, features_new)

            # 3. Chạy thuật toán Hungarian
            matches, unmatched_tracks, unmatched_detections = matching.linear_assignment(
                cost_matrix=cost_matrix,
                tracks=self.tracks,
                detections=valid_detections,
                max_distance=self.cosine_threshold
            )

        # XỬ LÝ 3 KẾT QUẢ TỪ THUẬT TOÁN HUNGARIAN:
        
        # Nhóm 1: Cập nhật các track đã ghép cặp thành công
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = valid_detections[det_idx]
            track.update(self.ukf_engine, det.bbox, det.feature)

        # Nhóm 2: Đánh dấu mất dấu đối với xe không tìm thấy
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Nhóm 3: Khởi tạo hồ sơ cho xe/người mới xuất hiện
        for det_idx in unmatched_detections:
            det = valid_detections[det_idx]
            if det.conf > 0.5: # Yêu cầu độ tự tin cao hơn để tạo ID mới (tránh nhiễu)
                if not self._is_duplicate_track(det):
                    self._initiate_track(det)

        # DỌN DẸP BỘ NHỚ
        # Xóa vĩnh viễn các track có state là DELETED
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Giới hạn số lượng hiển thị trên màn hình để tránh giật lag
        if len(self.tracks) > self.max_tracks:
            # Ưu tiên giữ: Đã xác nhận (Confirmed) -> Mới thấy (Update gần) -> Tồn tại lâu (Hits)
            self.tracks.sort(key=lambda t: (0 if t.is_confirmed() else 1, t.time_since_update, -t.hits, getattr(t, 'age', 0) * -1))
            self.tracks = self.tracks[:self.max_tracks]

        
    def _is_duplicate_track(self, det, iou_threshold=0.3, cos_sim_threshold=0.8):
        """
        Gating 3 lớp: Hình học (IoU) + Động học (Adaptive Maha) + Ngoại hình (EMA Cosine).
        SỬA: Truy xuất thông qua det.bbox thay vì det['bbox']
        """
        if not self.tracks:
            return False

        d_area = det.bbox[2] * det.bbox[3]
        if d_area <= 0: return False
        
        # Chuyển detection bbox sang chuẩn UKF [cx, cy, a, h]
        bbox_ukf = np.array([
            det.bbox[0] + det.bbox[2] / 2.0,
            det.bbox[1] + det.bbox[3] / 2.0,
            det.bbox[2] / max(det.bbox[3], 1e-3),
            det.bbox[3]
        ], dtype=np.float32)

        for t in self.tracks:
            # Chỉ check duplicate với các track vừa mới nhìn thấy (<= 2 frames)
            if t.is_confirmed() and t.time_since_update <= 2:
                t_bbox = t.to_tlwh()
                t_area = t_bbox[2] * t_bbox[3]
                
                # Tính Intersection nhanh
                x_left = max(det.bbox[0], t_bbox[0])
                y_top = max(det.bbox[1], t_bbox[1])
                x_right = min(det.bbox[0] + det.bbox[2], t_bbox[0] + t_bbox[2])
                y_bottom = min(det.bbox[1] + det.bbox[3], t_bbox[1] + t_bbox[3])
                
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
                        t_feat = t.get_feature() 
                        cos_sim = np.dot(t_feat, det.feature)
                        
                        # Nếu trùng khớp cả 3 yếu tố -> Khẳng định là Duplicate
                        if cos_sim > cos_sim_threshold:
                            return True
        return False

    def _initiate_track(self, det):
        """Tạo hồ sơ theo dõi mới (Track) cho vật thể."""
        bbox = det.bbox 
        bbox_ukf = np.array([
            bbox[0] + bbox[2] / 2.0,
            bbox[1] + bbox[3] / 2.0,
            bbox[2] / max(bbox[3], 1e-3),
            bbox[3]
        ], dtype=np.float32)

        mean, covariance = self.ukf_engine.initiate(bbox_ukf)
        
        # SỬA LỖI BIẾN: Truyền đúng biến next_id
        new_track = Track(
            mean=mean,
            covariance=covariance,
            track_id=self.next_id,
            class_id=det.class_id,
            feature=det.feature
        )
        
        # Áp đặt luật chơi về tuổi thọ cho track mới sinh ra
        new_track._max_age = self.max_age
        new_track._n_init = self.n_init
        
        self.tracks.append(new_track)
        self.next_id += 1

    def get_results(self):
        """Lấy danh sách các Track an toàn (Confirmed) để đem qua cho Visualizer vẽ."""
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