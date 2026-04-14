import cv2
import numpy as np

class SparseOpticalFlowCMC:
    def __init__(self, max_corners=200, quality_level=0.01, min_distance=30, scale_factor=0.5):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.scale_factor = scale_factor # Tối ưu 1: Tỉ lệ thu nhỏ ảnh
        
        self.prev_gray = None
        self.prev_pts = None

    def apply(self, current_frame):
        # 1. Downscale ảnh để tăng tốc độ tính toán
        small_frame = cv2.resize(current_frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        H = np.eye(2, 3, dtype=np.float32)

        if self.prev_gray is None:
            self._extract_new_pts(curr_gray)
            self.prev_gray = curr_gray
            return H

        # 2. Tính toán Optical Flow trên ảnh nhỏ
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, None)
            valid = (status == 1).ravel()
            
            if valid.sum() >= 4: # RANSAC cần ít nhất 4 điểm
                H_est, inliers = cv2.estimateAffinePartial2D(self.prev_pts[valid], curr_pts[valid], method=cv2.RANSAC)
                
                if H_est is not None: 
                    # --- TOÁN HỌC ---
                    # Bù trừ lại phần dịch chuyển (translation) về kích thước ảnh gốc
                    H_est[0, 2] /= self.scale_factor 
                    H_est[1, 2] /= self.scale_factor
                    H = H_est
                    
                # Giữ lại các điểm tốt (Inliers)
                self.prev_pts = curr_pts[valid][inliers.ravel() == 1].reshape(-1, 1, 2)
            else:
                self.prev_pts = None

        # 3. Tối ưu 2: Adaptive Refresh
        # Chỉ trích xuất thêm điểm mới nếu số điểm đang theo dõi rớt xuống dưới 40% (mất dấu quá nhiều)
        min_pts_threshold = int(self.max_corners * 0.4)
        if self.prev_pts is None or len(self.prev_pts) < min_pts_threshold:
            self._extract_new_pts(curr_gray, existing_pts=self.prev_pts)

        self.prev_gray = curr_gray
        return H

    def _extract_new_pts(self, gray_img, existing_pts=None):
        mask = np.ones_like(gray_img) * 255
        mask[int(gray_img.shape[0]*0.8):, :] = 0 # Bỏ nắp capo
        
        # Nếu đang có sẵn điểm, không tạo điểm mới đè lên vùng đó
        if existing_pts is not None and len(existing_pts) > 0:
            for pt in existing_pts:
                cv2.circle(mask, tuple(pt.ravel().astype(int)), self.min_distance, 0, -1)

        # Trích xuất thêm lượng điểm CÒN THIẾU, không phải lấy toàn bộ 200 điểm
        needed_corners = self.max_corners if existing_pts is None else self.max_corners - len(existing_pts)
        
        if needed_corners > 0:
            new_pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=needed_corners, 
                                            qualityLevel=self.quality_level, 
                                            minDistance=self.min_distance, mask=mask)
            if existing_pts is None:
                self.prev_pts = new_pts
            elif new_pts is not None:
                self.prev_pts = np.vstack((existing_pts, new_pts))