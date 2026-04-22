import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as FilterUKF
from filterpy.kalman import MerweScaledSigmaPoints
import cv2

class TrackUKF:
    """
    Bộ lọc Kalman Phi Tuyến (Unscented Kalman Filter) thiết kế riêng cho việc
    theo dõi mục tiêu 2D (Bounding Box) với mô hình động học Constant Turn Rate 
    and Velocity (CTRV) mở rộng.

    Trạng thái 8 chiều (8D State):
    [cx, cy, a, h, vx, vy, omega, gamma]
    Trong đó:
    - cx, cy: Tọa độ tâm hộp bao.
    - a: Aspect ratio (tỷ lệ chiều rộng / chiều cao).
    - h: Chiều cao hộp bao.
    - vx, vy: Vận tốc di chuyển theo trục x, y.
    - omega: Vận tốc góc (Tốc độ xoay hướng).
    - gamma: Tỷ lệ giãn nở khung hình (Perspective expansion rate).
    """

    def __init__(self, fps=10):
        """
        Khởi tạo bộ lọc UKF.

        Args:
            fps (int, optional): Số khung hình trên giây để tính bước nhảy thời gian dt.
                                 Mặc định là 10.
        """
        self.dt = 1.0 / float(fps)
        
        def transition_function(x, dt):
            """
            Hàm chuyển đổi trạng thái phi tuyến f(x).
            Dự đoán trạng thái của đối tượng ở bước thời gian tiếp theo (k+1).

            Args:
                x (numpy.ndarray): Vector trạng thái 8 chiều ở thời điểm k.
                dt (float): Bước nhảy thời gian.

            Returns:
                numpy.ndarray: Vector trạng thái 8 chiều dự đoán ở thời điểm k+1.
            """
            # [QUAN TRỌNG 1]: Giải nén phần tử thứ 8 thành biến `gamma` thay vì `vh`
            cx, cy, a, h, vx, vy, omega, gamma = x
            
            # Logic tính toán không gian 2D (Chống chia 0 khi xe đi thẳng)
            if abs(omega) < 1e-4:
                cx_new = cx + vx * dt
                cy_new = cy + vy * dt
                vx_new = vx
                vy_new = vy
            else:
                cx_new = cx + (vx / omega) * np.sin(omega * dt) - (vy / omega) * (1 - np.cos(omega * dt))
                cy_new = cy + (vx / omega) * (1 - np.cos(omega * dt)) + (vy / omega) * np.sin(omega * dt)
                
                vx_new = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
                vy_new = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
                
            # Aspect ratio (tỷ lệ khung hình) thay đổi rất chậm, có thể coi như hằng số
            a_new = a 
            
            # [QUAN TRỌNG 2]: Tính toán chiều cao mới dựa trên hàm mũ của gamma (Perspective Kinematics)
            h_new = h * np.exp(gamma * dt)
            
            # [QUAN TRỌNG 3]: Đóng gói trả về vector trạng thái mới, vị trí thứ 8 phải truyền `gamma`
            return np.array([cx_new, cy_new, a_new, h_new, vx_new, vy_new, omega, gamma], dtype=np.float64)

        def measurement_function(x):
            """
            Hàm đo lường phi tuyến h(x).
            Ánh xạ từ không gian trạng thái (State Space) sang không gian đo lường 
            (Measurement Space) mà camera nhìn thấy.

            Args:
                x (numpy.ndarray): Vector trạng thái 8 chiều.

            Returns:
                numpy.ndarray: Vector đo lường 4 chiều [cx_obs, cy_obs, a_obs, h_obs] 
                               đã được kẹp (clip) trong không gian độ phân giải của camera.
            """
            # Trạng thái x của bạn: [cx, cy, a, h, vx, vy, omega, gamma]
            cx, cy, a, h = x[0], x[1], x[2], x[3]
            
            # 1. Giải mã chiều rộng (w) từ aspect ratio (a)
            w = a * h
            
            # 2. Tính tọa độ hộp giới hạn (Bounding Box) ngoài đời thực
            xmin = cx - w / 2.0
            xmax = cx + w / 2.0
            ymin = cy - h / 2.0
            ymax = cy + h / 2.0
            
            # 3. Mô phỏng góc nhìn vật lý của Camera (Dataset KITTI: 1242x375)
            # YOLO sẽ không bao giờ nhìn thấy phần tọa độ âm hoặc vượt quá độ phân giải
            IMG_WIDTH = 1242.0
            IMG_HEIGHT = 375.0
            
            xmin_obs = max(0.0, xmin)
            xmax_obs = min(IMG_WIDTH, xmax)
            ymin_obs = max(0.0, ymin)
            ymax_obs = min(IMG_HEIGHT, ymax)
            
            # Nếu xe trôi hoàn toàn ra ngoài khung hình (tránh lỗi toán học chia 0)
            if xmin_obs >= xmax_obs or ymin_obs >= ymax_obs:
                # Trả về một tọa độ "ma" ở rất xa. 
                # DeepSORT tính Mahalanobis sẽ thấy vô lý và tự động Reject (đúng bản chất mất dấu)
                return np.array([-1000.0, -1000.0, a, h], dtype=np.float64)
                
            # 4. Tái cấu trúc lại hộp (Measurement) theo những gì YOLO thực sự thấy
            w_obs = xmax_obs - xmin_obs
            h_obs = ymax_obs - ymin_obs
            
            cx_obs = (xmin_obs + xmax_obs) / 2.0
            cy_obs = (ymin_obs + ymax_obs) / 2.0
            
            # Tính lại Aspect Ratio mới do hộp đã bị xén hẹp lại
            a_obs = w_obs / h_obs if h_obs > 0 else a
            
            return np.array([cx_obs, cy_obs, a_obs, h_obs], dtype=np.float64)
        # === KẾT THÚC PHẦN THAY THẾ ===

        # Tạo các điểm Sigma bằng thuật toán Merwe
        sigmas = MerweScaledSigmaPoints(n=8, alpha=0.3, beta=2., kappa=0)

        self.ukf = FilterUKF(
            dim_x=8, dim_z=4, dt=self.dt, 
            fx=transition_function, 
            hx=measurement_function, 
            points=sigmas
        )

    def _enforce_spd(self, matrix, min_eig=1e-5):
        """
        Đảm bảo ma trận luôn đối xứng và xác định dương (Symmetric Positive-Definite).
        Chống lại sai số dấu phẩy động (float precision errors) có thể làm vỡ bộ lọc UKF.

        Args:
            matrix (numpy.ndarray): Ma trận vuông (Thường là Covariance P, Q, hoặc R).
            min_eig (float, optional): Ngưỡng trị riêng nhỏ nhất. Mặc định là 1e-5.

        Returns:
            numpy.ndarray: Ma trận SPD an toàn.
        """
        matrix = (matrix + matrix.T) / 2.0
        try:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, min_eig)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T
        except:
            return matrix + np.eye(matrix.shape[0]) * min_eig

    def initiate(self, measurement):
        """
        Khởi tạo trạng thái ban đầu của bộ lọc khi quỹ đạo vừa được tạo.

        Args:
            measurement (numpy.ndarray): Vector đo lường ban đầu từ YOLO [cx, cy, a, h].
        """
        meas = np.asarray(measurement, dtype=np.float64).ravel()
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = meas
        
        h = meas[3]
        covariance = np.zeros((8, 8), dtype=np.float64)
        
        # 1. Độ bất định vị trí: Cho phép sai số nhỏ (tin tưởng hờ vào vị trí khởi tạo)
        covariance[0, 0] = covariance[1, 1] = covariance[3, 3] = (0.1 * h) ** 2
        covariance[2, 2] = 1e-2
        
        # Độ bất định vận tốc khởi tạo lớn (Chưa có lịch sử di chuyển)
        self.ukf.P[4, 4] = self.ukf.P[5, 5] = (10.0 * h) ** 2
        self.ukf.P[6, 6] = 1e-2
        
        # Tỷ lệ giãn nở (gamma) ban đầu chưa rõ, cho phép dao động trong khoảng 10%
        self.ukf.P[7, 7] = 1e-2
        
        self.ukf.x = mean
        self.ukf.P = self._enforce_spd(covariance)

    def _get_dynamic_Q(self, h):
        """
        Tính toán ma trận nhiễu hệ thống Q linh động dựa trên chiều cao vật thể.

        Args:
            h (float): Chiều cao hộp bao của vật thể.

        Returns:
            numpy.ndarray: Ma trận nhiễu Q (8x8) đã được đảm bảo SPD.
        """
        std_pos = 0.05 * h
        std_vel = 0.5 * h  # GIỮ NGUYÊN: Đủ lớn để hệ thống phản ứng với ma trận kéo lùi của CMC
        
        Q = np.zeros((8, 8), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = Q[3, 3] = std_pos ** 2
        Q[2, 2] = 1e-4
        
        # Sửa dòng gán nhiễu Q cho biến thứ 8 (index 7)
        Q[4, 4] = Q[5, 5] = std_vel ** 2
        Q[6, 6] = 1e-4
        # Nhiễu hệ thống của gamma: Cho phép tỷ lệ giãn nở thay đổi từ từ
        Q[7, 7] = 1e-4
        
        return self._enforce_spd(Q)

    def predict(self, H_camera=None):
        """
        Thực hiện bước dự đoán (Predict Step) của UKF.
        Áp dụng bù trừ chuyển động máy ảnh (CMC) trực tiếp vào vector trạng thái
        và ma trận hiệp phương sai trước khi dự đoán vật lý.

        Args:
            H_camera (numpy.ndarray, optional): Ma trận biến đổi Affine 2x3 từ CMC.
        """
        if H_camera is not None:
            # 1. Bù trừ Vị trí [cx, cy]
            pos = self.ukf.x[:2].reshape(1, 1, 2)
            self.ukf.x[:2] = cv2.transform(pos, H_camera).ravel()

            # Lấy ma trận R (xoay + giãn nở)
            R2x2 = H_camera[:, :2]
            scale = np.sqrt(abs(np.linalg.det(R2x2)))
            R_rot_only = R2x2 / (scale + 1e-6) 
            
            # 2. Bù trừ Vận tốc [vx, vy] chỉ bằng ma trận xoay
            self.ukf.x[4:6] = R_rot_only @ self.ukf.x[4:6]
            
            # 3. Bù trừ sự thay đổi kích thước do góc camera
            self.ukf.x[3] *= scale
            self.ukf.x[7] *= scale

            # 4. Truyền biến đổi CMC vào sự bất định (Covariance P)
            self.ukf.P[:2, :2] = R2x2 @ self.ukf.P[:2, :2] @ R2x2.T
            self.ukf.P[4:6, 4:6] = R_rot_only @ self.ukf.P[4:6, 4:6] @ R_rot_only.T

        # Thực thi Unscented Transform
        self.ukf.P = self._enforce_spd(self.ukf.P)
        self.ukf.Q = self._get_dynamic_Q(self.ukf.x[3])
        self.ukf.predict()
        self.ukf.P = self._enforce_spd(self.ukf.P)

    def update(self, measurement):
        """
        Thực hiện bước cập nhật (Update Step) của UKF.
        Điều chỉnh dự đoán dựa trên dữ liệu đo lường thực tế từ mạng Detection.

        Args:
            measurement (numpy.ndarray): Vector đo lường từ YOLO [cx, cy, a, h].
        """
        h = max(self.ukf.x[3], 1.0)
        
        # Tăng cực kỳ nhẹ nhàng từ 0.02 lên 0.03 để UKF hơi "lười" hơn một chút
        # (Chống lại hiện tượng rung viền Bounding Box của mạng AI)
        std_pos = 0.03 * h  
        std_h   = 0.03 * h  
        
        R = np.zeros((4, 4), dtype=np.float64)
        R[0, 0] = R[1, 1] = std_pos ** 2
        R[2, 2] = 1e-4      # Aspect ratio giữ nguyên
        R[3, 3] = std_h ** 2
        
        self.ukf.R = self._enforce_spd(R)
        self.ukf.update(np.asarray(measurement, dtype=np.float64))
        self.ukf.P = self._enforce_spd(self.ukf.P)
        
    @property
    def mean(self):
        """
        numpy.ndarray: Lấy vector trạng thái 8D hiện tại [cx, cy, a, h, vx, vy, omega, gamma].
        """
        return self.ukf.x

    @property
    def covariance(self):
        """
        numpy.ndarray: Lấy ma trận hiệp phương sai 8x8 hiện tại (Độ bất định).
        """
        return self.ukf.P