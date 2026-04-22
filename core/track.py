import numpy as np
from core.ukf import TrackUKF  

class TrackState:
    """
    Enum định nghĩa các trạng thái vòng đời của một quỹ đạo (Track).
    
    Attributes:
        TENTATIVE (int): Trạng thái tạm thời. Quỹ đạo mới được khởi tạo, chưa đủ độ tin cậy.
        CONFIRMED (int): Trạng thái đã xác nhận. Quỹ đạo đã liên tục được cập nhật trong nhiều khung hình.
        DELETED (int): Trạng thái bị xóa. Quỹ đạo đã mất dấu quá lâu và sẽ bị loại bỏ khỏi hệ thống.
    """
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

class Track:
    """
    Đại diện cho một quỹ đạo (Track) đang được theo dõi trong hệ thống.
    Quản lý trạng thái động học thông qua Unscented Kalman Filter (UKF) và 
    ngoại hình (ReID features) thông qua Exponential Moving Average (EMA).
    """

    def __init__(self, fps, track_id, class_id, bbox, feature):
        """
        Khởi tạo một đối tượng quỹ đạo mới.

        Args:
            fps (int): Tốc độ khung hình (Frame per second) của video/camera.
            track_id (int): Định danh duy nhất cho quỹ đạo này.
            class_id (int): Nhãn phân loại của đối tượng (VD: 0 cho Người, 2 cho Ô tô).
            bbox (numpy.ndarray): Tọa độ hộp bao ban đầu định dạng [x_top_left, y_top_left, width, height].
            feature (numpy.ndarray): Vector đặc trưng nhận dạng (ReID feature) ban đầu.
        """
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
        """
        Dự đoán trạng thái động học tiếp theo của đối tượng bằng bộ lọc UKF.

        Args:
            H_camera (numpy.ndarray, optional): Ma trận biến đổi Affine của khung hình 
                để bù trừ chuyển động máy ảnh (CMC). Mặc định là None.
        """
        self.ukf.predict(H_camera)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection_bbox, detection_feature, img_w, img_h):
        """
        Cập nhật trạng thái quỹ đạo dựa trên kết quả nhận diện (Detection) mới nhất.
        Thực hiện cả cập nhật động học (Vị trí, vận tốc) và cập nhật ngoại hình (Vector đặc trưng).

        Args:
            detection_bbox (numpy.ndarray): Tọa độ hộp bao nhận diện [x, y, w, h].
            detection_feature (numpy.ndarray): Vector đặc trưng từ mạng ReID.
            img_w (int): Chiều rộng của khung hình gốc.
            img_h (int): Chiều cao của khung hình gốc.
        """
        bbox_ukf = self._tlwh_to_xyah(detection_bbox)
        self.last_bbox = np.asarray(detection_bbox, dtype=np.float32).copy()
        
        current_w = self.ukf.mean[2] * self.ukf.mean[3]
        new_w = detection_bbox[2]
        
        # ==========================================
        # 1. LƯU LẠI VẬN TỐC CŨ (TRƯỚC KHI UPDATE)
        # ==========================================
        old_vx = self.ukf.mean[4]
        old_vy = self.ukf.mean[5]

        # ==========================================
        # 2. CHẠY UPDATE UKF BÌNH THƯỜNG 
        # (Chỉ giữ lại Soft Reboot cơ bản theo w)
        # ==========================================
        if new_w > 1.5 * current_w or new_w < 0.6 * current_w or self.time_since_update > 1:
            self.ukf.covariance[0, 0] = (2.0 * bbox_ukf[3]) ** 2
            self.ukf.covariance[1, 1] = (2.0 * bbox_ukf[3]) ** 2
            self.ukf.covariance[2, 2] = 1e-1           
            self.ukf.covariance[3, 3] = (1.0 * bbox_ukf[3]) ** 2  
            self.ukf.update(bbox_ukf)
        else:
            self.ukf.update(bbox_ukf)

        # ==========================================
        # 3. POST-UPDATE CONSTRAINING (RÀNG BUỘC VẬT LÝ SAU UPDATE)
        # ==========================================
        new_vx = self.ukf.mean[4]
        new_vy = self.ukf.mean[5]
        
        # A. Giới hạn Gia tốc (Max Acceleration)
        # Cài đặt ngưỡng: Xe không thể thay đổi quá 100 pixel mỗi frame
        max_dv = 100.0 
        
        dvx = new_vx - old_vx
        dvy = new_vy - old_vy
        
        if abs(dvx) > max_dv:
            # Ép vận tốc về ngưỡng tối đa cho phép
            self.ukf.mean[4] = old_vx + np.sign(dvx) * max_dv
            
        if abs(dvy) > max_dv:
            self.ukf.mean[5] = old_vy + np.sign(dvy) * max_dv

        # B. Chống lật hướng đột ngột (Directional Inertia)
        # Nếu xe đang có vận tốc rõ rệt (> 5) mà UKF lại tính ra vận tốc ngược chiều
        if abs(old_vx) > 5.0 and (old_vx * self.ukf.mean[4] < 0):
            # Ép xe "phanh gấp" (v = 0) thay vì bay ngược lại
            self.ukf.mean[4] = 0.0
            
        if abs(old_vy) > 5.0 and (old_vy * self.ukf.mean[5] < 0):
            self.ukf.mean[5] = 0.0

        # ==========================================
        # 4. CẬP NHẬT REID (BẢO VỆ MÉP MÀN HÌNH)
        # ==========================================
        if detection_feature is not None:
            # 1. Đưa tất cả vào trong block if để tránh lỗi UnboundLocalError
            x, y, w, h = detection_bbox
            margin = 15
            
            # Logic khóa không gian: Chống đầu độc EMA (Không cập nhật nếu vật thể chạm lề)
            is_safe_inside = (x > margin and y > margin and 
                              x + w < img_w - margin and 
                              y + h < img_h - margin)
                              
            if is_safe_inside:
                # 2. Chuẩn hóa vector đầu vào (L2 Normalization)
                det_norm = np.linalg.norm(detection_feature)
                if det_norm > 1e-6:
                    detection_feature /= det_norm
                    
                if self.smooth_feature is None:
                    # Dùng .copy() để tránh tham chiếu đè ô nhớ trong Python
                    self.smooth_feature = detection_feature.copy() 
                else:
                    # 3. Cập nhật EMA (Hệ số alpha = 0.9 nghĩa là bảo toàn 90% đặc trưng cũ)
                    alpha = 0.9 
                    self.smooth_feature = alpha * self.smooth_feature + (1 - alpha) * detection_feature
                    
                    # 4. Chuẩn hóa lại vector EMA sau khi cộng (Dùng max để chống chia cho 0)
                    self.smooth_feature /= max(np.linalg.norm(self.smooth_feature), 1e-6)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self):
        """
        Đánh dấu quỹ đạo đã bị mất dấu trong khung hình hiện tại.
        Xóa quỹ đạo nếu nó mới là TENTATIVE hoặc đã mất dấu vượt quá thời gian tối đa (max_age).
        """
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED
        
    def is_tentative(self): 
        """Kiểm tra xem quỹ đạo có đang ở trạng thái TENTATIVE hay không."""
        return self.state == TrackState.TENTATIVE
        
    def is_confirmed(self): 
        """Kiểm tra xem quỹ đạo có đang ở trạng thái CONFIRMED hay không."""
        return self.state == TrackState.CONFIRMED
        
    def is_deleted(self): 
        """Kiểm tra xem quỹ đạo có đang ở trạng thái DELETED hay không."""
        return self.state == TrackState.DELETED

    def get_features(self):
        """
        Trích xuất vector đặc trưng ngoại hình hiện tại của quỹ đạo.
        Gói feature duy nhất (smooth_feature) vào mảng 2D: shape (1, 512).
        Điều này giúp tương thích ngược với hàm cdist trong module matching.py.

        Returns:
            numpy.ndarray: Ma trận chứa vector đặc trưng đã làm mịn. Trả về mảng rỗng nếu không có.
        """
        if hasattr(self, 'smooth_feature') and self.smooth_feature is not None:
            return np.array([self.smooth_feature])
        return np.array([])

    def to_tlwh(self):
        """
        Chuyển đổi trạng thái trung bình (mean) của UKF từ dạng [cx, cy, a, h] 
        sang dạng [x_top_left, y_top_left, width, height].

        Returns:
            numpy.ndarray: Mảng chứa tọa độ hộp bao định dạng TLWH.
        """
        ret = self.ukf.mean[:4].copy()
        ret[2] *= ret[3] 
        ret[0] -= ret[2] / 2 
        ret[1] -= ret[3] / 2 
        return ret
    
    def _tlwh_to_xyah(self, tlwh):
        """
        Chuyển đổi tọa độ hộp bao từ dạng [x_top_left, y_top_left, width, height] 
        sang định dạng trạng thái [cx, cy, a, h] để UKF có thể xử lý.

        Args:
            tlwh (numpy.ndarray): Tọa độ hộp bao định dạng TLWH.

        Returns:
            numpy.ndarray: Mảng trạng thái định dạng [cx, cy, a, h].
        """
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2 
        ret[2] /= max(ret[3], 1e-3) 
        return ret