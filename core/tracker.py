import numpy as np
from core.track import Track, TrackState
from core import matching

class Detection:
    def __init__(self, bbox, conf, class_id, feature=None):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.conf = float(conf)
        self.class_id = int(class_id)
        self.feature = feature

class Tracker:
    def __init__(self, max_age=30, n_init=3, cosine_threshold=0.35, iou_threshold=0.7, fps=10):
        self.max_age = max_age
        self.n_init = n_init
        self.cosine_threshold = cosine_threshold
        self.iou_threshold = iou_threshold
        self.fps = fps
        self.tracks = []
        self.next_id = 1

    def update(self, bboxes, confs, class_ids, features, frame_shape, H_camera=None):
        img_h, img_w = frame_shape
        
        for trk in self.tracks:
            trk.predict(H_camera)
            x, y, w, h = trk.to_tlwh()
            
            # --- ĐÃ SỬA LẠI LOGIC LỀ ẢNH ---
            cx, cy = x + w / 2.0, y + h / 2.0
            
            # Điều kiện 1: Tâm xe văng hẳn ra ngoài camera -> Xóa ngay lập tức (không cần chờ)
            is_center_outside = (cx < 0 or cx > img_w or cy < 0 or cy > img_h)
            
            # Điều kiện 2: Xe chạm lề (còn trong camera nhưng dễ bị YOLO bắt hụt)
            margin = 15
            is_touching_margin = (x < margin or y < margin or x + w > img_w - margin or y + h > img_h - margin)
            
            if is_center_outside and trk.time_since_update > 0:
                trk.state = TrackState.DELETED
            elif is_touching_margin and trk.time_since_update > 5: # Cho phép YOLO miss 5 frame ở lề
                trk.state = TrackState.DELETED

        valid_dets = [Detection(bboxes[i], confs[i], class_ids[i], features[i] if len(features) > 0 else None) for i in range(len(bboxes))]
        unmatched_trks_idx = [i for i, t in enumerate(self.tracks) if not t.is_deleted()]
        unmatched_dets_idx = list(range(len(valid_dets)))

        # ====================================================================
        # VÒNG 1: CASCADE REID (Ưu tiên ngoại hình)
        # ====================================================================
        matched_v1 = []
        for level in range(1, self.max_age + 1):
            if not unmatched_dets_idx: break
            level_trk_idx = [i for i in unmatched_trks_idx if self.tracks[i].time_since_update == level]
            if not level_trk_idx: continue

            level_tracks = [self.tracks[i] for i in level_trk_idx]
            level_dets = [valid_dets[i] for i in unmatched_dets_idx]

            cost_matrix = matching.compute_cosine_distance(level_tracks, [d.feature for d in level_dets])
            for r, trk in enumerate(level_tracks):
                for c, det in enumerate(level_dets):
                    if trk.class_id != det.class_id or cost_matrix[r, c] > self.cosine_threshold:
                        cost_matrix[r, c] = 1e5

            matches, un_t, un_d = matching.linear_assignment(cost_matrix, level_tracks, level_dets, 1.0)
            for trk_i, det_i in matches:
                matched_v1.append((level_trk_idx[trk_i], unmatched_dets_idx[det_i]))
            unmatched_dets_idx = [unmatched_dets_idx[i] for i in un_d]

        for trk_i, det_i in matched_v1:
            self.tracks[trk_i].update(valid_dets[det_i].bbox, valid_dets[det_i].feature, img_w, img_h)
            unmatched_trks_idx.remove(trk_i)

        # ====================================================================
        # VÒNG 2: IOU MATCHING (Dùng box UKF dự đoán)
        # ====================================================================
        # ====================================================================
        # VÒNG 2: IOU MATCHING (Dùng box UKF dự đoán)
        # ====================================================================
        if len(unmatched_trks_idx) > 0 and len(unmatched_dets_idx) > 0:
            iou_trk_idx = [i for i in unmatched_trks_idx if self.tracks[i].time_since_update <= 2]
            iou_tracks = [self.tracks[i] for i in iou_trk_idx]
            iou_dets = [valid_dets[i] for i in unmatched_dets_idx]

            if len(iou_tracks) > 0:
                iou_cost = matching.compute_iou_matrix(iou_tracks, iou_dets, img_w, img_h)
                
                # --- THÊM MỚI: CHỐT CHẶN REID ---
                # Tính trước ma trận ngoại hình để làm lưới lọc
                feature_cost_v2 = matching.compute_cosine_distance(iou_tracks, [d.feature for d in iou_dets])

                for r, trk in enumerate(iou_tracks):
                    for c, det in enumerate(iou_dets):
                        # Lưới lọc: Khác Class HOẶC Ngoại hình quá khác biệt (Cos > 0.55) -> Vô hiệu hóa ghép tọa độ!
                        # Ngưỡng 0.55 lỏng hơn Vòng 1 (0.35) để linh hoạt, nhưng đủ chặt để phân biệt Đen - Đỏ.
                        if trk.class_id != det.class_id or feature_cost_v2[r, c] > 0.55:
                            iou_cost[r, c] = 1e5

                matches_v2, un_t_iou, un_d_iou = matching.linear_assignment(iou_cost, iou_tracks, iou_dets, self.iou_threshold)

                for trk_i, det_i in matches_v2:
                    actual_trk_idx = iou_trk_idx[trk_i]
                    actual_det_idx = unmatched_dets_idx[det_i]
                    det = valid_dets[actual_det_idx]
                    if det.feature is None: det.feature = np.zeros(512, dtype=np.float32)
                    self.tracks[actual_trk_idx].update(det.bbox, det.feature, img_w, img_h)
                    unmatched_trks_idx.remove(actual_trk_idx)
                    
                unmatched_dets_idx = [unmatched_dets_idx[i] for i in un_d_iou]

        # ====================================================================
        # VÒNG 3: VƯỢT RÀO KALMAN BẰNG IOM (Dùng hộp UKF đã bù trừ CMC)
        # ====================================================================
        if len(unmatched_trks_idx) > 0 and len(unmatched_dets_idx) > 0:
            
            # Cho phép quét tất cả các xe đang mất dấu (Bỏ giới hạn time_since_update)
            rec_trk_idx = [i for i in unmatched_trks_idx] 
            rec_tracks = [self.tracks[i] for i in rec_trk_idx]
            rec_dets = [valid_dets[i] for i in unmatched_dets_idx]

            if len(rec_tracks) > 0:
                bypass_cost = np.full((len(rec_tracks), len(rec_dets)), 1e5, dtype=np.float32)
                
                # Tính trước ma trận ngoại hình (ReID) để làm chốt chặn
                feature_cost_v3 = matching.compute_cosine_distance(rec_tracks, [d.feature for d in rec_dets])
                
                for r, trk in enumerate(rec_tracks):
                    
                    # QUAN TRỌNG: Dùng trực tiếp hộp UKF vì nó đã được 
                    # cộng dồn CMC hoàn hảo qua từng frame bị che khuất
                    proj_x, proj_y, proj_w, proj_h = trk.to_tlwh()
                    proj_area = proj_w * proj_h

                    for c, det in enumerate(rec_dets):
                        if trk.class_id != det.class_id: continue

                        det_x, det_y, det_w, det_h = det.bbox
                        det_area = det_w * det_h

                        # Tính diện tích đè lên nhau
                        ix1, iy1 = max(proj_x, det_x), max(proj_y, det_y)
                        ix2, iy2 = min(proj_x+proj_w, det_x+det_w), min(proj_y+proj_h, det_y+det_h)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                        # Tính IoM (Intersection over Minimum)
                        iom = inter / max(min(proj_area, det_area), 1e-6)

                        # ĐIỀU KIỆN KÉP: Tọa độ lọt vào nhau > 60% VÀ Ngoại hình giống nhau
                        # Ngưỡng cosine 0.6 nới lỏng chút ít cho góc chụp thay đổi
                        if iom > 0.6 and feature_cost_v3[r, c] <= 0.6: 
                            bypass_cost[r, c] = 1.0 - iom

                matches_v3, un_t_v3, un_d_v3 = matching.linear_assignment(bypass_cost, rec_tracks, rec_dets, 1.0)

                for trk_i, det_i in matches_v3:
                    actual_trk_idx = rec_trk_idx[trk_i]
                    actual_det_idx = unmatched_dets_idx[det_i]
                    
                    det = valid_dets[actual_det_idx]
                    if det.feature is None: det.feature = np.zeros(512, dtype=np.float32)
                    self.tracks[actual_trk_idx].update(det.bbox, det.feature, img_w, img_h)
                    unmatched_trks_idx.remove(actual_trk_idx)
                    
                unmatched_dets_idx = [unmatched_dets_idx[i] for i in un_d_v3]

        # ====================================================================
        # QUẢN LÝ VÒNG ĐỜI
        # ====================================================================
        for det_idx in unmatched_dets_idx:
            det = valid_dets[det_idx]
            if det.conf > 0.5:
                new_track = Track(self.fps, self.next_id, det.class_id, det.bbox, det.feature)
                self.tracks.append(new_track)
                self.next_id += 1

        for track_idx in unmatched_trks_idx:
            trk = self.tracks[track_idx]
            x, y, w, h = trk.to_tlwh()
            margin = 15
            
            # CHỈ XÓA khi xe thực sự unmatch (time_since_update > 0) 
            # VÀ nó đang nằm sát lề (có nguy cơ đi ra khỏi camera)
            is_out_of_frame = (x < margin or y < margin or x + w > img_w - margin or y + h > img_h - margin)
            
            if is_out_of_frame:
                trk.state = TrackState.DELETED
            else:
                trk.mark_missed()

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def get_results(self):
        return [(t.track_id, t.class_id, *t.to_tlwh()) for t in self.tracks if t.time_since_update == 0 and t.is_confirmed()]