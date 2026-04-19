import numpy as np
import cv2
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

def fx(x, dt):
    """ Mô hình động học Constant Turn Rate and Velocity (CTRV) """
    x_out = np.copy(x)
    cx, cy, a, h, vx, vy, omega, vh = x
    
    if abs(omega) < 1e-4:
        x_out[0] = cx + vx * dt
        x_out[1] = cy + vy * dt
    else:
        x_out[0] = cx + (vx * np.sin(omega * dt) - vy * (1 - np.cos(omega * dt))) / omega
        x_out[1] = cy + (vx * (1 - np.cos(omega * dt)) + vy * np.sin(omega * dt)) / omega
        
    x_out[3] = h + vh * dt
    # Vận tốc xoay chuyển theo tốc độ góc omega
    x_out[4] = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
    x_out[5] = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
    return x_out

def hx(x):
    # Trích xuất không gian đo lường [cx, cy, a, h]
    return x[:4]

class TrackUKF:
    def __init__(self, measurement):
        points = MerweScaledSigmaPoints(n=8, alpha=0.1, beta=2., kappa=-5)
        self.ukf = UnscentedKalmanFilter(dim_x=8, dim_z=4, dt=1.0, hx=hx, fx=fx, points=points)
        
        self.ukf.x = np.zeros(8)
        self.ukf.x[:4] = measurement
        
        h = max(measurement[3], 10.0)
        # Giảm quy mô nhiễu khởi tạo vận tốc để chống vọt lố (Overshoot)
        self.ukf.P = np.diag([
            (0.1 * h)**2, (0.1 * h)**2, 1e-2, (0.1 * h)**2,
            (0.05 * h)**2, (0.05 * h)**2, 1e-4, (0.01 * h)**2
        ])

    @property
    def x(self):
        return self.ukf.x
        
    @property
    def P(self):
        return self.ukf.P

    def _enforce_spd(self, matrix):
        """ Ép buộc ma trận về trạng thái Symmetric Positive Definite """
        matrix = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _get_dynamic_Q(self, x):
        h = max(x[3], 10.0)
        # Nhiễu hệ thống Q tỷ lệ với độ lớn của vật thể
        Q = np.diag([
            (0.05 * h)**2, (0.05 * h)**2, 1e-4, (0.05 * h)**2,
            (0.02 * h)**2, (0.02 * h)**2, 1e-4, (0.01 * h)**2
        ])
        return Q

    def predict(self, H_camera=None):
        if H_camera is not None:
            # BoT-SORT CMC Integration [12]
            M = H_camera[:, :2]  # Rotation / Scale Matrix 2x2
            T = H_camera[:, 2]   # Translation vector 2x1

            # 1. Biến đổi Vị trí Trung tâm [cx, cy]
            pos = self.ukf.x[:2].reshape(1, 1, 2)
            self.ukf.x[:2] = cv2.transform(pos, H_camera).ravel()

            # 2. Biến đổi Vận tốc [vx, vy] (Tuyệt đối không cộng thêm vector Tịnh tiến T)
            self.ukf.x[4:6] = M @ self.ukf.x[4:6]

            # 3. Biến đổi Toàn vẹn Ma trận Hiệp phương sai P (8x8)
            # Thay vì chỉ cập nhật khối 2x2 như cũ, phải tái cấu trúc chéo để không phá vỡ liên kết vật lý
            M_8x8 = np.eye(8)
            M_8x8[0:2, 0:2] = M
            M_8x8[4:6, 4:6] = M
            self.ukf.P = M_8x8 @ self.ukf.P @ M_8x8.T

        self.ukf.P = self._enforce_spd(self.ukf.P)
        self.ukf.Q = self._get_dynamic_Q(self.ukf.x)
        self.ukf.predict()
        self.ukf.P = self._enforce_spd(self.ukf.P)

    def update(self, measurement, confidence=1.0):
        h = max(self.ukf.x[3], 10.0)
        std_pos = 0.05 * h
        
        # KIẾN TRÚC NSA-KALMAN :
        # Nếu YOLO rất tự tin (confidence tiến gần 1), noise_scale tiến về 0.05, 
        # bộ lọc sẽ bị ép phải cập nhật trạng thái sát với bounding box thực tế.
        noise_scale = max(1.0 - confidence, 0.05)
        
        R = np.diag([
            (std_pos * noise_scale)**2, 
            (std_pos * noise_scale)**2, 
            (1e-2 * noise_scale)**2, 
            (std_pos * noise_scale)**2
        ])
        
        self.ukf.R = self._enforce_spd(R)
        self.ukf.update(np.asarray(measurement, dtype=np.float64))
        self.ukf.P = self._enforce_spd(self.ukf.P)