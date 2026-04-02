# Class quản lý 1 chiếc xe (ID, tuổi thọ, EMA feature).
from cv2.gapi import mean
import numpy as np

class TrackState:
    """
    Định nghĩa 3 trạng thái của một đối tượng đang được theo dõi:
    - TENTATIVE: Vừa xuất hiện, đang chờ theo dõi thêm để xác nhận không phải nhiễu.
    - CONFIRMED: Đã bám track ổn định, được phép hiển thị lên màn hình.
    - DELETED: Bị mất dấu quá lâu, chuẩn bị xóa khỏi bộ nhớ.
    """
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

    class Track:
        """
        Hồ sơ của một đối tượng đang được theo dõi.
        """
        def __init__(self, mean, covariance, track_id, class_id, feature):\
        # Định danh
        self.track_id = track_id
        self.class_id = class_id

        # Trạng thái động học
        self.mean = mean    # Vector [cx, cy, a, h, vx, vy, va, vh]
        self.covariance = covariance # Ma trận hiệp phương sai 8x8

        # Trạng thái ngoại hình
        self.feature = feature  # Vector đặc trưng ngoại hình 512 chiều
        self.alpha = 0.9  # Hệ số EMA: 90% trọng số cũ, 10% trọng số mới

        # Quản lý vòng đời
        self.hits = 1 # Số frame đã match thành công
        self.age = 1 # Số frame từ khi track được tạo
        self.time_since_update = 0 # Số frame bị mất dấu liên tiếp
        self.state = TrackState.TENTATIVE # Trạng thái ban đầu là TENTATIVE

        # Ngưỡng cấu hình
        self.n_init = 3 # Số frame cần match để chuyển sang CONFIRMED
        self.max_age = 30 # Số frame tối đa được phép mất dấu trước khi chuyển sang DELETED

    def predict(self, kf):
        """
        Dự đoán vị trí tiếp theo của track dựa trên ukf.
        Liên tục cập nhật mỗi frame, ngay cả khi không có detections nào match được.
        """
        # Cập nhật thông số bằng ukf
        self.mean, self.covariance = ukf.predict(self.mean, self.covariance)

        # Tăng tuổi thọ và thời gian mất dấu
        self.age += 1
        self.time_since_update += 1

    def update(self, ukf, detection_bbox, dectection_feature):
        """
        Cập nhật track khi có một detection match thành công.
        """

        # Đổi bounding box sang định dạng [cx, cy, a, h]
        bbox_ukf = self._tlwh_to_xyah(detection_bbox)

        # Cập nhật thông số bằng ukf
        self.mean, self.covariance = ukf.update(self.mean, self.covariance, bbox_ukf)

        # Cập nhật đặc trưng ngoại hình bằng EMA
        self.feature = self.alpha * self.feature + (1 - self.alpha) * detection_feature

        # Chuẩn hóa L2
        self.feature /= np.linalg.norm(self.feature)

        # Reset bộ đếm mất dấu
        self.hits += 1
        self.time_since_update = 0

        # Cập nhật trạng thái nếu đủ điều kiện
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """
        Xử lý khi không tìm thấy đối tượng trong frame hiện tại.
        """
        # Nếu đang ở trạng thái TENTATIVE, chuyển ngay sang DELETED vì chưa đủ tin cậy.
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        # Nếu đang ở trạng thái CONFIRMED, chỉ chuyển sang DELETED nếu đã mất dấu quá lâu.
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
        
    def is_tentative(self):
        return self.state == TrackState.TENTATIVE
    
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self):
        return self.state == TrackState.DELETED
    
    def to_tlwh(self):
        """
        Dịch chuyển từ định dạng [cx, cy, a, h] sang [x, y, w, h] để hiển thị.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3] # w = a * h
        ret[0] -= ret[2] / 2 # x_min = cx - w/2
        ret[1] -= ret[3] / 2 # y_min = cy - h/2
        return ret
    
    def _tlwh_to_xyah(self, tlwh):
        """
        Chuyển đổi bounding box từ định dạng [x, y, w, h] sang [cx, cy, a, h].
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2 # cx = x_min + w/2, cy = y_min + h/2
        ret[2] /= ret[3] # a = w / h
        return ret