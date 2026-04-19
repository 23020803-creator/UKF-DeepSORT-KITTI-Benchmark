import numpy as np
from core.ukf import TrackUKF

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

class Track:
    def __init__(self, tlwh, score, track_id, class_id, n_init, max_age, feature=None):
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self._n_init = n_init
        self._max_age = max_age
        
        self.feature = feature
        self.smooth_feature = feature
        
        xyah = self._tlwh_to_xyah(tlwh)
        self.ukf = TrackUKF(xyah)

    def predict(self, H_camera=None):
        self.ukf.predict(H_camera=H_camera)
        self.age += 1
        self.time_since_update += 1

    def update(self, tlwh, score, feature=None):
        self.hits += 1
        self.time_since_update = 0
        
        # Chuyển trạng thái an toàn
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

        xyah = self._tlwh_to_xyah(tlwh)
        
        # Cập nhật động học kết hợp NSA-Kalman Filter
        self.ukf.update(xyah, confidence=score)

        # Cập nhật đặc trưng ngoại hình (EMA Feature Bank) 
        if feature is not None:
            if self.smooth_feature is None:
                self.smooth_feature = feature
            else:
                # Hệ số momentum alpha = 0.9 để thiên vị lịch sử và khử nhiễu nhất thời
                alpha = 0.9
                self.smooth_feature = alpha * self.smooth_feature + (1.0 - alpha) * feature
                self.smooth_feature /= np.linalg.norm(self.smooth_feature)

    def mark_missed(self):
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_tentative(self): return self.state == TrackState.TENTATIVE
    def is_confirmed(self): return self.state == TrackState.CONFIRMED
    def is_deleted(self): return self.state == TrackState.DELETED

    def _tlwh_to_xyah(self, tlwh):
        """ Chuyển đổi từ [top_left_x, top_left_y, width, height] sang [center_x, center_y, aspect_ratio, height] """
        ret = np.asarray(tlwh).copy()
        ret[0] += ret[2] / 2.0  # center_x = top_left_x + width/2
        ret[1] += ret[3] / 2.0  # center_y = top_left_y + height/2
        ret[2] /= ret[3]        # aspect_ratio = width / height
        return ret