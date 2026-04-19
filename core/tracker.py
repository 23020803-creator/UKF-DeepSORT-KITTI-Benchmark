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

    def update(self, bboxes, confs, class_ids, features, frame_shape, H_camera=None, frame_idx=0):
        img_h, img_w = frame_shape
        
        # ====================================================================
        # BƯỚC 0: CẬP NHẬT QUỸ ĐẠO DỰ ĐOÁN (KALMAN FILTER PREDICT)
        # ====================================================================
        for trk in self.tracks:
            trk.predict(H_camera)
            # Lấy tọa độ hộp dự đoán của Tracker (Dùng biến x, y, w, h)
            x, y, w, h = trk.to_tlwh()
            
            cx, cy = x + w / 2.0, y + h / 2.0
            
            # Điều kiện 1: Xe đã văng HOÀN TOÀN ra khỏi không gian camera (Không còn dính 1 pixel nào)
            is_completely_outside = (x + w < 0 or x > img_w or y + h < 0 or y > img_h)
            if is_completely_outside:
                trk.state = TrackState.DELETED
                continue
            
            # Điều kiện 2: Xe vẫn còn dính ở lề (Tâm văng ra ngoài hoặc chạm margin)
            # YOLO rất hay bắt hụt ở đây. Ta cấp cho nó "kim bài miễn tử" 15 frame!
            margin = 15
            is_touching_margin = (x < margin or y < margin or x + w > img_w - margin or y + h > img_h - margin)
            is_center_outside = (cx < 0 or cx > img_w or cy < 0 or cy > img_h)
            
            if (is_touching_margin or is_center_outside) and trk.time_since_update > 15:
                trk.state = TrackState.DELETED

        # ====================================================================
        # BỘ LỌC CẮT TỈA (EDGE-NOISE PRUNING) - Lọc rác YOLO
        # ====================================================================
        valid_dets = []
        for i in range(len(bboxes)):
            # Tọa độ gốc của YOLO
            x_det, y_det, w_det, h_det = bboxes[i]
            
            margin_prune = 15
            is_touching_edge = (x_det <= margin_prune or x_det + w_det >= img_w - margin_prune)
            
            # Nếu hộp tọa độ CHẠM LỀ và CHIỀU RỘNG BỊ TEO QUÁ NHỎ (< 45 pixel) -> Cắt bỏ
            if is_touching_edge and w_det < 60:
                continue 
                
            feature = features[i] if len(features) > 0 else None
            valid_dets.append(Detection(bboxes[i], confs[i], class_ids[i], feature))

        unmatched_trks_idx = [i for i, t in enumerate(self.tracks) if not t.is_deleted()]
        unmatched_dets_idx = list(range(len(valid_dets)))

        # ====================================================================
        # VÒNG 1: CASCADE REID (Ưu tiên ngoại hình nhưng có CHỐT CHẶN KHÔNG GIAN)
        # ====================================================================
        matched_v1 = []
        for level in range(1, self.max_age + 1):
            if not unmatched_dets_idx: break
            level_trk_idx = [i for i in unmatched_trks_idx if self.tracks[i].time_since_update == level]
            if not level_trk_idx: continue

            level_tracks = [self.tracks[i] for i in level_trk_idx]
            level_dets = [valid_dets[i] for i in unmatched_dets_idx]

            # Tính ma trận ngoại hình (ReID)
            cost_matrix = matching.compute_cosine_distance(level_tracks, [d.feature for d in level_dets])
            iou_matrix = matching.compute_iou_matrix(level_tracks, level_dets, img_w, img_h)
            
            # GỌI MAHALANOBIS MỚI TẠO TẠI ĐÂY
            maha_matrix = matching.compute_mahalanobis_distance(level_tracks, level_dets)

            for r, trk in enumerate(level_tracks):
                for c, det in enumerate(level_dets):
                    # 1. BỘ LỌC CƠ BẢN: Khác Class hoặc Ngoại hình quá khác biệt
                    if trk.class_id != det.class_id or cost_matrix[r, c] > self.cosine_threshold:
                        cost_matrix[r, c] = 1e5
                        continue

                    # =================================================================
                    # 2. BỘ LỌC ĐỘNG HỌC MAHALANOBIS (DYNAMIC KINEMATIC GATING)
                    # =================================================================
                    # Ngưỡng Chi-square cho 4 bậc tự do (cx, cy, a, h) tại p=0.05 là 9.4877
                    gating_threshold = 9.4877
                    dynamic_threshold = gating_threshold
                    
                    # DYNAMIC GATING: Nới lỏng giới hạn vật lý nhờ ReID
                    # cost_matrix[r, c] lúc này đang lưu giá trị Cosine Distance của ReID
                    if cost_matrix[r, c] < 0.2:
                        # Xe giống hệt nhau -> Nới lỏng giới hạn không gian lên gấp 3 lần 
                        # để cho phép quỹ đạo bị lệch do đi qua gốc cây/vật cản
                        dynamic_threshold = gating_threshold * 3.0
                    
                    if maha_matrix[r, c] > dynamic_threshold:
                        # Cấm ghép cặp (Gating)
                        cost_matrix[r, c] = 1e5
                        continue

                    # Nâng cao: Blend Cost. Giúp Hungarian có cái nhìn toàn diện hơn
                    # Ưu tiên: Ngoại hình giống + Động học (Mahalanobis) hợp lý + IoU cao
                    # Đưa maha_matrix về scale [0, 1] xấp xỉ để blend
                    norm_maha = min(maha_matrix[r, c] / gating_threshold, 1.0)
                    cost_matrix[r, c] = 0.7 * cost_matrix[r, c] + 0.15 * iou_matrix[r, c] + 0.15 * norm_maha

            # Chạy thuật toán Hungarian trên ma trận đã được thêm các "chốt chặn"
            matches, un_t, un_d = matching.linear_assignment(cost_matrix, level_tracks, level_dets, 1.0)

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
                
                # Tính trước ma trận ngoại hình để làm lưới lọc
                feature_cost_v2 = matching.compute_cosine_distance(iou_tracks, [d.feature for d in iou_dets])

                for r, trk in enumerate(iou_tracks):
                    for c, det in enumerate(iou_dets):
                        # Vòng 2 phải cực kỳ khắt khe: Ngoại hình khác biệt (Cos > 0.55) -> Vô hiệu hóa!
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

                        # Lấy tọa độ thực tế của hộp YOLO thay vì hộp dự đoán
                        # ---------------------------------------------------------
                        # BƯỚC 1: TÍNH TOÁN CÁC THÔNG SỐ KHÔNG GIAN (BẮT BUỘC)
                        # ---------------------------------------------------------
                        # 1. Tính toán IoM như cũ
                        det_x, det_y, det_w, det_h = det.bbox
                        proj_area = proj_w * proj_h
                        det_area = det_w * det_h
                        
                        inter_x1 = max(proj_x, det_x)
                        inter_y1 = max(proj_y, det_y)
                        inter_x2 = min(proj_x + proj_w, det_x + det_w)
                        inter_y2 = min(proj_y + proj_h, det_y + det_h)
                        
                        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        iom = inter / max(min(proj_area, det_area), 1e-6)

                        # =======================================================
                        # [CHỐT CHẶN 1]: CENTER SHIFT GATING (Chống nhận vơ)
                        # =======================================================
                        det_cx, det_cy = det_x + det_w / 2, det_y + det_h / 2
                        proj_cx, proj_cy = proj_x + proj_w / 2, proj_y + proj_h / 2
                        dist_x = abs(det_cx - proj_cx)
                        dist_y = abs(det_cy - proj_cy)
                            
                        # Bản chất: Nếu xe A bị xén ở lề, tâm hộp YOLO chỉ lệch tối đa 
                        # khoảng 1 nửa chiều rộng (0.5 * proj_w). Nếu lệch tới > 0.8, 
                        # nghĩa là có một Xe B khác đang nằm ở mép bên kia của vùng dự đoán.
                        if dist_x > proj_w * 0.8 or dist_y > proj_h * 0.8:
                            continue # Block ngay lập tức, không cho cướp ID

                        # --- Phân loại khu vực ---
                        is_near_left = (det_x <= 30)
                        is_near_right = (det_x + det_w >= img_w - 30)
                        is_near_top = (det_y <= 30)
                        is_near_bottom = (det_y + det_h >= img_h - 30)
                        is_near_edge = is_near_left or is_near_right or is_near_top or is_near_bottom

                        # =======================================================
                        # [CHỐT CHẶN 2]: INVARIANT EDGE ALIGNMENT & DYNAMIC GATING
                        # =======================================================
                        if not is_near_edge:
                            # Ở giữa ảnh: Cần ReID xác nhận chống ID Switch
                            if iom > 0.60 and feature_cost_v3[r, c] <= 0.60: 
                                bypass_cost[r, c] = 1.0 - iom
                        else:
                            # Ở lề ảnh: REID ĐÃ CHẾT, CHỈ TIN TOÁN HỌC CẠNH (EDGE)
                            # 1. Hạ IoM xuống 0.50 để bù sai số YOLO bắt muộn
                            if iom > 0.50: 
                                edge_penalty = 1.0 
                                
                                # Tính toán độ lệch của cạnh đối diện lề bị xén
                                if is_near_left:
                                    edge_penalty = abs((proj_x + proj_w) - (det_x + det_w)) / (proj_w + 1e-6)
                                elif is_near_right:
                                    edge_penalty = abs(proj_x - det_x) / (proj_w + 1e-6)
                                elif is_near_top:
                                    edge_penalty = abs((proj_y + proj_h) - (det_y + det_h)) / (proj_h + 1e-6)
                                elif is_near_bottom:
                                    edge_penalty = abs(proj_y - det_y) / (proj_h + 1e-6)
                                    
                                # 2. NGƯỠNG ĐỘNG (Dynamic Threshold) bù đắp điểm mù khi xe đánh lái
                                # Tăng 3% sai số cho mỗi frame mất dấu. Tối đa 40%.
                                dynamic_edge_thresh = min(0.40, 0.15 + 0.03 * trk.time_since_update)
                                
                                # 3. Chốt chặn sinh tử nhận thức độ bất định
                                if edge_penalty > dynamic_edge_thresh:
                                    continue
                                    
                                # Vượt qua bài test -> Ép cost cực thấp để ghép cặp bằng mọi giá
                                bypass_cost[r, c] = 0.5 * (1.0 - iom) + 0.5 * edge_penalty

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
        # QUẢN LÝ VÒNG ĐỜI (Đã vá lỗi Logic sát lề)
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
            
            # 1. Luôn tăng bộ đếm time_since_update trước
            trk.mark_missed()
            
            is_out_of_frame = (x < margin or y < margin or x + w > img_w - margin or y + h > img_h - margin)
            
            # 2. Sửa lỗi cốt lõi: Chỉ giết ID nếu nó sát lề VÀ đã mất dấu quá 15 frame
            # (Đồng bộ với logic 15 frame ở đầu hàm update)
            if is_out_of_frame and trk.time_since_update > 15:
                trk.state = TrackState.DELETED
            # 3. Hoặc nếu xe nằm giữa màn hình nhưng mất dấu vượt quá max_age
            elif trk.is_deleted(): 
                pass
                
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def get_results(self):
        # Trả về tuple 10 phần tử: (track_id, class_id, ukf_x, ukf_y, ukf_w, ukf_h, yolo_x, yolo_y, yolo_w, yolo_h)
        results = []
        for t in self.tracks:
            # Cho phép UKF duy trì dự đoán trên màn hình tối đa 10 frame khi mất dấu
            if t.time_since_update <= 10 and t.is_confirmed():
                
                # LOGIC LỌC BÓNG MA YOLO:
                if t.time_since_update == 0:
                    yolo_box = t.last_bbox # YOLO bắt được -> Lấy tọa độ thật
                else:
                    yolo_box = [0, 0, 0, 0] # YOLO mù -> Ép tọa độ YOLO về 0 để tàng hình
                    
                results.append((t.track_id, t.class_id, *t.to_tlwh(), *yolo_box))
                
        return results