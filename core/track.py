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

        self.smooth_feature = None
        if feature is not None:
            self.smooth_feature = feature.copy()
        self.alpha = 0.1

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
        
        current_w = self.ukf.mean[2] * self.ukf.mean[3]
        new_w = detection_bbox[2]
        
        # Đổi thành > 1. Xe track bình thường sau khi predict() sẽ có giá trị là 1.
        # > 1 nghĩa là xe đã bị mất dấu ít nhất 1 frame (ví dụ: lấp sau cây).
        if new_w > 1.5 * current_w or new_w < 0.6 * current_w or self.time_since_update > 1:
            
            # 1. REBOOT TOÀN BỘ BỘ LỌC UKF
            # Dùng initiate() để thiết lập lại tâm, kích thước và ma trận hiệp phương sai P từ đầu.
            # Thao tác này chặn hoàn toàn lỗi Overshoot của hàm update().
            self.ukf.initiate(bbox_ukf)
            
            # 2. TẨY NÃO ĐẶC TRƯNG REID
            if detection_feature is not None:
                self.smooth_feature = detection_feature.copy()
                
        else:
            # 3. CHỈ gọi update khi xe không bị khuất và kích thước bình thường
            self.ukf.update(bbox_ukf)

        # ==========================================
        # Cập nhật Feature EMA (Lọc thông thấp)
        # ==========================================
        if detection_feature is not None:
            x, y, w, h = detection_bbox
            margin = 15
            is_truncated = (x < margin or y < margin or (x + w) > img_w - margin or (y + h) > img_h - margin)
            
            if not is_truncated:
                norm = np.linalg.norm(detection_feature)
                if norm > 1e-6:
                    detection_feature /= norm
                
                if not hasattr(self, 'smooth_feature') or self.smooth_feature is None:
                    self.smooth_feature = detection_feature
                else:
                    self.smooth_feature = 0.1 * detection_feature + 0.9 * self.smooth_feature
                    self.smooth_feature /= max(np.linalg.norm(self.smooth_feature), 1e-6)

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
    
    # TÌM VÀ XÓA DÒNG NÀY:
    # def get_features(self): return np.array(self.features)

    # THAY THẾ BẰNG ĐOẠN SAU:
    def get_features(self):
        """
        Gói feature duy nhất (smooth_feature) vào mảng 2D: shape (1, 512).
        Điều này giúp tương thích ngược với hàm cdist trong matching.py
        """
        if hasattr(self, 'smooth_feature') and self.smooth_feature is not None:
            return np.array([self.smooth_feature])
        return np.array([])

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