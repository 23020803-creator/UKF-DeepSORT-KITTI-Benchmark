import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as FilterUKF
from filterpy.kalman import MerweScaledSigmaPoints
import cv2

class TrackUKF:
    def __init__(self, fps=10):
        self.dt = 1.0 / float(fps)
        
        def transition_function(x, dt):
            # Trạng thái x: [cx, cy, a, h, vx, vy, omega, vh]
            cx, cy, a, h, vx, vy, omega, vh = x
            
            # Tránh chia cho 0 khi xe đi thẳng (omega quá nhỏ)
            if abs(omega) < 1e-4:
                cx_new = cx + vx * dt
                cy_new = cy + vy * dt
                vx_new = vx
                vy_new = vy
            else:
                # Quỹ đạo vòng cung (Khúc cua)
                cx_new = cx + (vx / omega) * np.sin(omega * dt) - (vy / omega) * (1 - np.cos(omega * dt))
                cy_new = cy + (vx / omega) * (1 - np.cos(omega * dt)) + (vy / omega) * np.sin(omega * dt)
                
                # Vận tốc bị xoay đi một góc omega * dt
                vx_new = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
                vy_new = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
                
            # Các thông số kích thước hộp (a, h) giả định tuyến tính
            a_new = a
            h_new = h + vh * dt
            
            return np.array([cx_new, cy_new, a_new, h_new, vx_new, vy_new, omega, vh], dtype=np.float64)

        def measurement_function(x):
            return x[:4]

        sigmas = MerweScaledSigmaPoints(n=8, alpha=0.3, beta=2., kappa=0)

        self.ukf = FilterUKF(
            dim_x=8, dim_z=4, dt=self.dt, 
            fx=transition_function, 
            hx=measurement_function, 
            points=sigmas
        )

    def _enforce_spd(self, matrix, min_eig=1e-5):
        matrix = (matrix + matrix.T) / 2.0
        try:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, min_eig)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T
        except:
            return matrix + np.eye(matrix.shape[0]) * min_eig

    def initiate(self, measurement):
        meas = np.asarray(measurement, dtype=np.float64).ravel()
        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = meas
        
        h = meas[3]
        covariance = np.zeros((8, 8), dtype=np.float64)
        covariance[0, 0] = covariance[1, 1] = covariance[3, 3] = (0.05 * h) ** 2
        covariance[2, 2] = 1e-4
        covariance[4, 4] = covariance[5, 5] = covariance[7, 7] = (0.01 * h) ** 2
        covariance[6, 6] = 1e-5
        
        self.ukf.x = mean
        self.ukf.P = self._enforce_spd(covariance)

    def _get_dynamic_Q(self, h):
        std_pos = 0.1 * h
        std_vel = 0.05 * h
        
        Q = np.zeros((8, 8), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = Q[3, 3] = std_pos ** 2
        Q[2, 2] = 1e-4
        Q[4, 4] = Q[5, 5] = Q[7, 7] = std_vel ** 2
        Q[6, 6] = 1e-5
        return self._enforce_spd(Q)

    def predict(self, H_camera=None):
        if H_camera is not None:
            pos = self.ukf.x[:2].reshape(1, 1, 2)
            self.ukf.x[:2] = cv2.transform(pos, H_camera).ravel()

            R2x2 = H_camera[:, :2]
            scale = np.sqrt(abs(np.linalg.det(R2x2)))
            R_rot_only = R2x2 / (scale + 1e-6) 
            
            self.ukf.x[4:6] = R_rot_only @ self.ukf.x[4:6]
            
            self.ukf.x[3] *= scale
            self.ukf.x[7] *= scale

            self.ukf.P[:2, :2] = R2x2 @ self.ukf.P[:2, :2] @ R2x2.T
            self.ukf.P[4:6, 4:6] = R_rot_only @ self.ukf.P[4:6, 4:6] @ R_rot_only.T

        self.ukf.P = self._enforce_spd(self.ukf.P)
        self.ukf.Q = self._get_dynamic_Q(self.ukf.x[3])
        self.ukf.predict()
        self.ukf.P = self._enforce_spd(self.ukf.P)

    def update(self, measurement):
        h = max(self.ukf.x[3], 1.0)
        std_pos = 0.02 * h
        
        R = np.zeros((4, 4), dtype=np.float64)
        R[0, 0] = R[1, 1] = R[3, 3] = std_pos ** 2
        R[2, 2] = 1e-4
        
        self.ukf.R = self._enforce_spd(R)
        self.ukf.update(np.asarray(measurement, dtype=np.float64))
        self.ukf.P = self._enforce_spd(self.ukf.P)
        
    @property
    def mean(self):
        return self.ukf.x

    @property
    def covariance(self):
        return self.ukf.P