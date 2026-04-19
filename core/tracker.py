import numpy as np
from core import matching
from core.track import Track, TrackState

class Detection:
    def __init__(self, tlwh, conf, class_id, feature=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.conf = float(conf)
        self.class_id = int(class_id)
        self.feature = feature
        self.xyah = self._tlwh_to_xyah(self.tlwh)
        
    def _tlwh_to_xyah(self, tlwh):
        """ Chuyển đổi từ [top_left_x, top_left_y, width, height] sang [center_x, center_y, aspect_ratio, height] """
        ret = np.asarray(tlwh).copy()
        ret[0] += ret[2] / 2.0  # center_x = top_left_x + width/2
        ret[1] += ret[3] / 2.0  # center_y = top_left_y + height/2
        ret[2] /= ret[3]        # aspect_ratio = width / height
        return ret

class Tracker:
    def __init__(self, max_age=30, n_init=3, cosine_threshold=0.35, high_thresh=0.6):
        self.max_age = max_age
        self.n_init = n_init
        self.cosine_threshold = cosine_threshold
        self.high_thresh = high_thresh
        self.tracks = []
        self._next_id = 1

    def update(self, valid_bboxes, valid_confs, valid_class_ids, features, frame_shape, H_camera=None, frame_idx=0):
        # Bước 0: Giai đoạn Dự Đoán (Predict Stage) với UKF + CMC
        for trk in self.tracks:
            trk.predict(H_camera=H_camera)

        # Chuyển đổi dữ liệu thô sang đối tượng Detection
        detections = []
        for bbox, conf, cid, feat in zip(valid_bboxes, valid_confs, valid_class_ids, features):
            detections.append(Detection(bbox, conf, cid, feat))

        # Phân loại Detections theo logic ByteTrack 
        det_high = [d for d in detections if d.conf >= self.high_thresh]
        det_low = [d for d in detections if d.conf < self.high_thresh]

        unmatched_trks_idx = list(range(len(self.tracks)))
        matched_v1 = []

        # --- VÒNG 1: Đối khớp High Confidence Detections (ReID + IoU + Mahalanobis) ---
        if len(det_high) > 0 and len(unmatched_trks_idx) > 0:
            trks_for_v1 = [self.tracks[i] for i in unmatched_trks_idx]
            
            cos_dist = matching.compute_cosine_distance(trks_for_v1, det_high)
            maha_dist = matching.compute_mahalanobis_distance(trks_for_v1, det_high)
            
            # Đóng cổng Mahalanobis: Chặn các kết nối có sự vi phạm quỹ đạo quá lớn
            cos_dist[maha_dist > 9.4877] = np.inf
            
            # Dung hợp ma trận chi phí theo kiểu BoT-SORT
            cost_matrix = matching.fuse_score(cos_dist, trks_for_v1, det_high)
            
            matches, un_trks, un_dets = matching.linear_assignment(
                cost_matrix, trks_for_v1, det_high, max_distance=0.8
            )
            
            # KHẮC PHỤC LỖI LOGIC: Lưu giữ đúng chỉ số thực của Track và Detection
            for t_idx, d_idx in matches:
                matched_v1.append((unmatched_trks_idx[t_idx], d_idx))
            
            unmatched_trks_idx = [unmatched_trks_idx[i] for i in un_trks]
            unmatched_det_high_idx = un_dets.tolist()
        else:
            unmatched_det_high_idx = list(range(len(det_high)))

        # Thực thi cập nhật cho các Track đã khớp tại Vòng 1
        for trk_idx, det_idx in matched_v1:
            det = det_high[det_idx]
            self.tracks[trk_idx].update(det.tlwh, det.conf, det.feature)

        # --- VÒNG 2: Đối khớp Low Confidence Detections (Chỉ sử dụng IoU) ---
        # Nhằm duy trì các track đang bị che khuất cục bộ (occlusion)
        matched_v2 = []
        if len(det_low) > 0 and len(unmatched_trks_idx) > 0:
            # Lọc lại: Chỉ ưu tiên các track vừa mới bị mất dấu (thời gian <= 2 khung hình)
            trks_for_v2_idx = [i for i in unmatched_trks_idx if self.tracks[i].time_since_update <= 2]
            trks_for_v2 = [self.tracks[i] for i in trks_for_v2_idx]
            
            if len(trks_for_v2) > 0:
                iou_matrix = matching.compute_iou_matrix(trks_for_v2, det_low)
                matches_v2, un_trks_v2, un_dets_v2 = matching.linear_assignment(
                    iou_matrix, trks_for_v2, det_low, max_distance=0.5
                )
                
                for t_idx, d_idx in matches_v2:
                    actual_trk_idx = trks_for_v2_idx[t_idx]
                    matched_v2.append((actual_trk_idx, d_idx))
                    unmatched_trks_idx.remove(actual_trk_idx)
                    
        # Cập nhật cho Vòng 2, bỏ qua đặc trưng ReID vì Low-Conf thường là nhiễu
        for trk_idx, det_idx in matched_v2:
            det = det_low[det_idx]
            self.tracks[trk_idx].update(det.tlwh, det.conf, feature=None)

        # --- Giai đoạn dọn dẹp & Khởi tạo ---
        for trk_idx in unmatched_trks_idx:
            self.tracks[trk_idx].mark_missed()

        # Chỉ khởi tạo ID mới từ các High Confidence Detections (Chống sinh rác IDSW)
        for det_idx in unmatched_det_high_idx:
            self._initiate_track(det_high[det_idx])

        # Bộ dọn dẹp: Hủy các track vượt quá max_age hoặc bị rác
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _initiate_track(self, detection):
        new_track = Track(
            detection.tlwh, detection.conf, self._next_id, 
            detection.class_id, self.n_init, self.max_age, detection.feature
        )
        self.tracks.append(new_track)
        self._next_id += 1

    def get_results(self):
        results = []
        for trk in self.tracks:
            # Chỉ trả về các Track đã được xác nhận (CONFIRMED) và hiện diện trong khung hình hiện tại
            if trk.is_confirmed() and trk.time_since_update <= 1:
                bbox = trk.ukf.x[:4]
                tlwh = matching._xyah_to_tlwh(bbox)
                results.append((tlwh, trk.track_id, trk.class_id))
        return results