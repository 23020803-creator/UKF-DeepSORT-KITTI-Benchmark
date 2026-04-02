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
    Nơi này chỉ lưu trữ dữ liệu, không chứa logic Deep Learning.
    """
    def __init__(self, mean, covariance, track_id, class_id, feature):
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

        # Ngưỡng cấu hình (Sẽ được tracker.py tự động map sang khi tạo mới)
        self._n_init = 3 # Số frame cần match để chuyển sang CONFIRMED
        self._max_age = 30 # Số frame tối đa được phép mất dấu trước khi chuyển sang DELETED

    def predict(self, kf):
        """
        Dự đoán vị trí tiếp theo của track dựa trên Bộ lọc UKF.
        Liên tục cập nhật mỗi frame, ngay cả khi không có detections nào match được.
        """
        # SỬA: Gọi đúng tên biến kf (Kalman Filter) được truyền vào từ tracker.py
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

        # Tăng tuổi thọ và thời gian mất dấu
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection_bbox, detection_feature):
        """
        Cập nhật track khi có một detection match thành công từ YOLO.
        """
        # Đổi bounding box sang định dạng [cx, cy, a, h] để phù hợp với toán học của UKF
        bbox_ukf = self._tlwh_to_xyah(detection_bbox)

        # Cập nhật thông số vị trí bằng UKF (SỬA: gọi đúng biến kf)
        self.mean, self.covariance = kf.update(self.mean, self.covariance, bbox_ukf)

        # Cập nhật đặc trưng ngoại hình bằng EMA (Exponential Moving Average)
        self.feature = self.alpha * self.feature + (1 - self.alpha) * detection_feature

        # BẮT BUỘC: Chuẩn hóa L2 lại cho Vector sau khi cộng EMA
        norm = np.linalg.norm(self.feature)
        if norm > 1e-6:
            self.feature /= norm

        # Reset bộ đếm mất dấu vì vừa tìm lại được đối tượng
        self.hits += 1
        self.time_since_update = 0

        # Nếu đang ở trạng thái chờ xác nhận mà đủ số hits (n_init), chuyển sang CONFIRMED
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """
        Xử lý khi không tìm thấy đối tượng trong frame hiện tại (Bị che khuất, đi khỏi cam).
        """
        # Nếu đang ở trạng thái TENTATIVE, chuyển ngay sang DELETED vì chưa đủ độ tin cậy.
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        # Nếu đang ở trạng thái CONFIRMED, chỉ chuyển sang DELETED nếu đã mất dấu quá số frame tối đa (max_age).
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED
        
    def is_tentative(self):
        return self.state == TrackState.TENTATIVE
    
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self):
        return self.state == TrackState.DELETED
    
    def get_feature(self):
        """
        [THÊM MỚI] Hàm cung cấp vector đặc trưng cho việc tính khoảng cách Cosine bên tracker.py
        """
        return self.feature

    def to_tlwh(self):
        """
        Dịch chuyển từ định dạng Toán học [cx, cy, a, h] trả ngược về [x_min, y_min, w, h] 
        để Visualizer vẽ Bounding Box.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3] # w = a * h
        ret[0] -= ret[2] / 2 # x_min = cx - w/2
        ret[1] -= ret[3] / 2 # y_min = cy - h/2
        return ret
    
    def _tlwh_to_xyah(self, tlwh):
        """
        Chuyển đổi bounding box từ định dạng vẽ [x_min, y_min, w, h] sang [cx, cy, a, h].
        """
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2 # cx = x_min + w/2, cy = y_min + h/2
        
        # SỬA LỖI TOÁN HỌC: Tránh lỗi Division by Zero (Chia cho 0) khi h = 0
        ret[2] /= max(ret[3], 1e-3) # a = w / h
        return ret