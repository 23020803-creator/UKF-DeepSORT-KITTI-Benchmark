import numpy as np
from core.ukf import TrackUKF  

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

class Track:
    def __init__(self, fps, track_id, class_id, bbox, feature):
        self.track_id = track_id
        self.class_id = class_id

        self.ukf = TrackUKF(fps=fps)
        bbox_ukf = self._tlwh_to_xyah(bbox)
        self.ukf.initiate(bbox_ukf)

        self.last_bbox = np.asarray(bbox, dtype=np.float32).copy()

        # SỬA: Dùng list để thiết lập "Mỏ neo", cấm dùng deque
        self.features = [] 
        if feature is not None:
            self.features.append(feature)

        self.hits = 1 
        self.age = 1 
        self.time_since_update = 0 
        self.state = TrackState.TENTATIVE 
        self._n_init = 3 
        self._max_age = 30 

    def predict(self, H_camera=None):
        self.ukf.predict(H_camera)
        if self.time_since_update > 0:
            self.ukf.mean[4:] *= 0.95 
        self.age += 1
        self.time_since_update += 1

    def update(self, detection_bbox, detection_feature, img_w, img_h):
        bbox_ukf = self._tlwh_to_xyah(detection_bbox)

        self.last_bbox = np.asarray(detection_bbox, dtype=np.float32).copy()
        
        # --- SỬA: ÉP UKF NHẢY CÓC KÍCH THƯỚC (CHỐNG LỰC Ỳ) ---
        current_w = self.ukf.mean[2] * self.ukf.mean[3]
        new_w = detection_bbox[2]
        
        # Nếu xe đột ngột nở to > 1.5 lần hoặc co lại < 0.6 lần,
        # Reset toàn bộ trạng thái để UKF quên đi quá khứ sai lệch!
        if new_w > 1.5 * current_w or new_w < 0.6 * current_w:
            # 1. Ghi đè toàn bộ: Tâm (cx, cy) và Kích thước (a, h)
            self.ukf.mean[:4] = bbox_ukf[:4]
            
            # 2. Reset vận tốc (vx, vy, omega, vh) về 0.
            self.ukf.mean[4:] = 0.0
            
            # 3. Phá vỡ Lực ỳ: Bơm lại độ bất định (Nhiễu) vào Ma trận Hiệp phương sai P
            # Lưu ý: Phải gọi self.ukf.ukf.P vì TrackUKF là class bọc ngoài
            h = max(bbox_ukf[3], 1.0)
            self.ukf.ukf.P[0, 0] = self.ukf.ukf.P[1, 1] = self.ukf.ukf.P[3, 3] = (0.05 * h) ** 2
            self.ukf.ukf.P[2, 2] = 1e-4
            self.ukf.ukf.P[4, 4] = self.ukf.ukf.P[5, 5] = self.ukf.ukf.P[7, 7] = (0.01 * h) ** 2
            self.ukf.ukf.P[6, 6] = 1e-5
            
            # Chống lỗi toán học ma trận
            self.ukf.ukf.P = self.ukf._enforce_spd(self.ukf.ukf.P)
            
        self.ukf.update(bbox_ukf)

        if detection_feature is not None:
            x, y, w, h = detection_bbox
            margin = 15
            is_truncated = (x < margin or y < margin or (x + w) > img_w - margin or (y + h) > img_h - margin)
            
            if not is_truncated:
                norm = np.linalg.norm(detection_feature)
                if norm > 1e-6:
                    detection_feature /= norm
                
                if len(self.features) < 50:
                    self.features.append(detection_feature)
                else:
                    self.features.pop(0) # Xóa ảnh cũ nhất (cái đuôi xe)
                    self.features.append(detection_feature) # Thêm ảnh mới nhất (toàn bộ xe)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED
        
    def is_tentative(self): return self.state == TrackState.TENTATIVE
    def is_confirmed(self): return self.state == TrackState.CONFIRMED
    def is_deleted(self): return self.state == TrackState.DELETED
    
    def get_features(self): return np.array(self.features)

    def to_tlwh(self):
        ret = self.ukf.mean[:4].copy()
        ret[2] *= ret[3] 
        ret[0] -= ret[2] / 2 
        ret[1] -= ret[3] / 2 
        return ret
    
    def _tlwh_to_xyah(self, tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2 
        ret[2] /= max(ret[3], 1e-3) 
        return ret