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
            margin = 15
            # Lỗi Sát thủ lề ảnh: Chỉ xóa bóng ma khi đã mất dấu
            if (x < margin or y < margin or x + w > img_w - margin or y + h > img_h - margin) and trk.time_since_update > 1:
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
        if len(unmatched_trks_idx) > 0 and len(unmatched_dets_idx) > 0:
            iou_trk_idx = [i for i in unmatched_trks_idx if self.tracks[i].time_since_update <= 2]
            iou_tracks = [self.tracks[i] for i in iou_trk_idx]
            iou_dets = [valid_dets[i] for i in unmatched_dets_idx]

            if len(iou_tracks) > 0:
                iou_cost = matching.compute_iou_matrix(iou_tracks, iou_dets, img_w, img_h)
                for r, trk in enumerate(iou_tracks):
                    for c, det in enumerate(iou_dets):
                        if trk.class_id != det.class_id:
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
        # VÒNG 3: VƯỢT RÀO KALMAN (BYPASS UKF) - CHÌA KHÓA TRỊ GIÃN NỞ
        # ====================================================================
        if len(unmatched_trks_idx) > 0 and len(unmatched_dets_idx) > 0:
            rec_trk_idx = [i for i in unmatched_trks_idx if self.tracks[i].time_since_update <= 2]
            rec_tracks = [self.tracks[i] for i in rec_trk_idx]
            rec_dets = [valid_dets[i] for i in unmatched_dets_idx]

            if len(rec_tracks) > 0:
                bypass_cost = np.full((len(rec_tracks), len(rec_dets)), 1e5, dtype=np.float32)
                import cv2 # Đảm bảo cv2 có sẵn trong scope này
                
                for r, trk in enumerate(rec_tracks):
                    
                    # 1. Lấy BBox thực tế tĩnh của frame trước (Tuyệt đối không lấy UKF)
                    last_x, last_y, last_w, last_h = trk.last_bbox
                    
                    # 2. Tự tay bù trừ chuyển động Camera (CMC)
                    if H_camera is not None:
                        # Tính tâm hộp cũ
                        cx = last_x + last_w / 2.0
                        cy = last_y + last_h / 2.0
                        
                        # Dịch chuyển tâm theo ma trận rung/tiến của Camera
                        pos = np.array([[[cx, cy]]], dtype=np.float32)
                        transformed_pos = cv2.transform(pos, H_camera).ravel()
                        new_cx, new_cy = transformed_pos[0], transformed_pos[1]
                        
                        # Tính lại tỷ lệ phình to/thu nhỏ của khung hình nếu có
                        R2x2 = H_camera[:, :2]
                        scale = np.sqrt(abs(np.linalg.det(R2x2)))
                        proj_w = last_w * scale
                        proj_h = last_h * scale
                        
                        # Áp lại thành dạng top-left width height
                        proj_x = new_cx - proj_w / 2.0
                        proj_y = new_cy - proj_h / 2.0
                    else:
                        proj_x, proj_y, proj_w, proj_h = last_x, last_y, last_w, last_h

                    proj_area = proj_w * proj_h

                    for c, det in enumerate(rec_dets):
                        if trk.class_id != det.class_id: continue

                        det_x, det_y, det_w, det_h = det.bbox
                        det_area = det_w * det_h

                        # 3. Tính diện tích đè lên nhau (Intersection)
                        ix1, iy1 = max(proj_x, det_x), max(proj_y, det_y)
                        ix2, iy2 = min(proj_x+proj_w, det_x+det_w), min(proj_y+proj_h, det_y+det_h)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                        # 4. Cứu vớt bằng IoM 
                        iom = inter / max(min(proj_area, det_area), 1e-6)

                        # Hạ ngưỡng xuống 0.6 vì chúng ta đang dịch chuyển manual
                        if iom > 0.6: 
                            bypass_cost[r, c] = 0.1

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
            self.tracks[track_idx].mark_missed()

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def get_results(self):
        return [(t.track_id, t.class_id, *t.to_tlwh()) for t in self.tracks if t.time_since_update == 0 and t.is_confirmed()]