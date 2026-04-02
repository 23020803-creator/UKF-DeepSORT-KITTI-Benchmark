import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as FilterUKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import cho_factor, cho_solve

class UKF:
    """
    Bộ lọc Unscented Kalman Filter .
    """
    def __init__(self):
        # Cấp phát sẵn ma trận F. 
        self._base_F = np.eye(8, dtype=np.float32)
        # Định nghĩa các hàm chuyển đổi.
        # Hàm chuyển đổi trạng thái.
        def transition_function(x, dt):
            F = self._base_F.copy()
            F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = float(dt)
            return F @ x

        # Hàm đo lường trích xuất [cx, cy, a, h] từ trạng thái 8 chiều.
        def measurement_function(x):
            return x[:4]

        # Khởi tạo Sigma Points.
        # alpha, beta, kappa là các thông số chuẩn cho phân phối Gaussian.
        sigmas = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=0)

        # Khởi tạo UKF.
        self.ukf = FilterUKF(
            dim_x=8, dim_z=4, dt=1.0, 
            fx=transition_function, 
            hx=measurement_function, 
            points=sigmas
        )

        self.ukf.x = np.zeros(8, dtype=np.float32)
        self.ukf.P = np.eye(8, dtype=np.float32)

        # Ma trận Nhiễu Hệ thống (Base Q)
        # Mô phỏng Constant Velocity Model: Nhiễu sinh ra từ Gia tốc (Acceleration)
        sigma_accel = 0.05  
        self._base_Q = np.zeros((8, 8), dtype=np.float32)
        dt = 1.0
        dt2, dt3, dt4 = dt**2, dt**3, dt**4

        # Block Matrix mô tả sự tương quan giữa Vị trí và Vận tốc
        block = np.array([
            [dt4/4, dt3/2],
            [dt3/2, dt2]
        ], dtype=np.float32) * (sigma_accel ** 2)

        for i in range(4):
            self._base_Q[i, i] = block[0, 0]
            self._base_Q[i, i+4] = block[0, 1]
            self._base_Q[i+4, i] = block[1, 0]
            self._base_Q[i+4, i+4] = block[1, 1]
        
        #  Ma trận nhiễu đo lường (Measurement Noise)
        # Aspect ratio (a) rất ổn định, gán nhiễu thấp (0.01) để tránh biến dạng bbox
        self._base_R = np.diag([1.0, 1.0, 0.01, 1.0]).astype(np.float32)
        self.ukf.R = self._base_R.copy()
        
        self.max_va = 2.0   # Tốc độ biến thiên aspect ratio tối đa
        self.max_vh = 100.0 # Tốc độ biến thiên height tối đa

    def _get_dynamic_Q(self, h):
        """
        Scale nhiễu theo chiều cao vật thể.
        Object ở xa (h nhỏ) -> nhiễu nhỏ; Object ở gần (h lớn) -> nhiễu lớn.
        """
        scale = max(h, 1.0) / 100.0
        Q_scaled = self._base_Q.copy()
        
        Q_scaled[:4, :4] *= scale
        # Tăng nhiễu vận tốc lên 0.5 để Tracker nhạy hơn khi rẽ
        Q_scaled[4:, 4:] *= scale * 0.5 
        return Q_scaled

    def initiate(self, measurement):
        """
        Khởi tạo trạng thái ban đầu cho một Track mới.
        """
        meas = np.asarray(measurement, dtype=np.float32).ravel()
        
        # Lọc rác từ YOLO
        if meas.shape[0] != 4 or not np.all(np.isfinite(meas)):
            meas = np.zeros(4, dtype=np.float32)

        meas[2] = max(meas[2], 1e-3)
        meas[3] = max(meas[3], 1e-3)

        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = meas
        
        h = meas[3]
        covariance = np.eye(8, dtype=np.float32)
        # Khởi tạo độ bất định ban đầu (P) tỷ lệ thuận theo chiều cao
        covariance[:4, :4] *= (0.05 * h) ** 2
        covariance[4:, 4:] *= (0.05 * 10 * h) ** 2 
        
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Dự đoán vị trí khung hình tiếp theo.
        """
        self.ukf.x = mean.astype(np.float64)
        self.ukf.P = covariance.astype(np.float64)
        
        # Scale Q động dựa trên trạng thái hiện tại
        h = max(self.ukf.x[3], 1.0)
        self.ukf.Q = self._get_dynamic_Q(h).astype(np.float64)
        
        self.ukf.predict()
        
        # Ép P luôn đối xứngxứng.
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2.0
        
        # h, a phải > 0.
        self.ukf.x[2] = max(self.ukf.x[2], 1e-3)
        self.ukf.x[3] = max(self.ukf.x[3], 1e-3)
        
        # Kẹp vận tốc để tránh Tracker trôi dạt quá xa khi mất dấudấu.
        self.ukf.x[6] = np.clip(self.ukf.x[6], -self.max_va, self.max_va)
        self.ukf.x[7] = np.clip(self.ukf.x[7], -self.max_vh, self.max_vh)
        
        return self.ukf.x.copy().astype(np.float32), self.ukf.P.copy().astype(np.float32)

    def gating_distance(self, mean, covariance, measurement):
        """
         Tính khoảng cách không gian dựa trên độ bất định.
        Sử dụng Phân rã Cholesky để tăng tốc O(N^3) -> O(N^2) và đảm bảo ổn định số học.
        """
        meas = np.asarray(measurement, dtype=np.float32).ravel()
        if meas.shape[0] != 4 or not np.all(np.isfinite(meas)):
            return float('inf') 
            
        d = meas - mean[:4]
        predicted_h = max(mean[3], 1.0)
        R_dynamic = self._base_R * predicted_h
        
        # Tính ma trận hiệp phương sai đổi mới (Innovation Covariance) S
        S = covariance[:4, :4] + R_dynamic
        S = (S + S.T) / 2.0 
        
        # Thêm Jitter để tránh lỗi ma trận suy biến (Singular Matrix)
        S += np.eye(4, dtype=np.float32) * 1e-6 
        
        try:
            c, low = cho_factor(S)
            return d.T @ cho_solve((c, low), d)
        except np.linalg.LinAlgError:
            # Fallback an toàn chặn ghép cặp nếu Toán học thất bại
            return float('inf')

    def update(self, mean, covariance, measurement):
        """
        Cập nhật tọa độ thực tế từ YOLO vào Track.
        """
        meas = np.asarray(measurement, dtype=np.float32).ravel()
        if meas.shape[0] != 4 or not np.all(np.isfinite(meas)):
            return mean.copy(), covariance.copy()

        meas[2] = max(meas[2], 1e-3)
        meas[3] = max(meas[3], 1e-3)

        self.ukf.x = mean.astype(np.float64)
        self.ukf.P = covariance.astype(np.float64)
        
        # Đồng bộ logic Dynamic R với hàm Gating.
        predicted_h = max(mean[3], 1.0)
        self.ukf.R = (self._base_R * predicted_h).astype(np.float64)
        
        self.ukf.update(meas.astype(np.float64))
        
        # Lặp lại các bước bảo vệ sau khi Update
        self.ukf.P = (self.ukf.P + self.ukf.P.T) / 2.0
        self.ukf.x[2] = max(self.ukf.x[2], 1e-3)
        self.ukf.x[3] = max(self.ukf.x[3], 1e-3)
        self.ukf.x[6] = np.clip(self.ukf.x[6], -self.max_va, self.max_va)
        self.ukf.x[7] = np.clip(self.ukf.x[7], -self.max_vh, self.max_vh)
        
        return self.ukf.x.copy().astype(np.float32), self.ukf.P.copy().astype(np.float32)