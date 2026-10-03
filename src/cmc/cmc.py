import cv2
import numpy as np

class SparseOpticalFlowCMC:
    """
    Lớp xử lý bù trừ chuyển động máy ảnh (Camera Motion Compensation - CMC) 
    sử dụng thuật toán Sparse Optical Flow (Lucas-Kanade) và RANSAC.
    """

    def __init__(self, max_corners=200, quality_level=0.01, min_distance=30, scale_factor=0.5):
        """
        Khởi tạo các tham số cho bộ ước lượng Optical Flow.

        Args:
            max_corners (int): Số lượng điểm đặc trưng (corners) tối đa để theo dõi.
            quality_level (float): Ngưỡng chất lượng tối thiểu để chấp nhận một góc (Shi-Tomasi).
            min_distance (int): Khoảng cách Euclid tối thiểu giữa các điểm đặc trưng.
            scale_factor (float): Hệ số thu nhỏ ảnh để tăng tốc độ xử lý (Tối ưu hiệu năng).
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.scale_factor = scale_factor
        
        self.prev_gray = None
        self.prev_pts = None

    def apply(self, current_frame):
        """
        Ước lượng ma trận biến đổi (Affine) giữa khung hình trước đó và khung hình hiện tại.

        Args:
            current_frame (numpy.ndarray): Khung hình BGR hiện tại.

        Returns:
            numpy.ndarray: Ma trận biến đổi Affine [2x3] (H). 
                           Trả về ma trận đơn vị (Identity Matrix) nếu không đủ điểm để ước lượng.
        """
        # BƯỚC 1: Tiền xử lý - Thu nhỏ ảnh và chuyển đổi sang không gian xám
        small_frame = cv2.resize(current_frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        H = np.eye(2, 3, dtype=np.float32)

        # Khởi tạo điểm đặc trưng cho khung hình đầu tiên
        if self.prev_gray is None:
            self._extract_new_pts(curr_gray)
            self.prev_gray = curr_gray
            return H

        # BƯỚC 2: Theo dõi điểm đặc trưng và tính toán ma trận tịnh tiến
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, None)
            valid = (status == 1).ravel()
            
            # Thuật toán RANSAC yêu cầu tối thiểu 4 cặp điểm hợp lệ
            if valid.sum() >= 4: 
                H_est, inliers = cv2.estimateAffinePartial2D(self.prev_pts[valid], curr_pts[valid], method=cv2.RANSAC)
                
                if H_est is not None: 
                    # Bù trừ hệ số tỷ lệ (scale_factor) cho các vector dịch chuyển (translation)
                    H_est[0, 2] /= self.scale_factor 
                    H_est[1, 2] /= self.scale_factor
                    H = H_est
                    
                # Lọc và cập nhật tập hợp các điểm thuộc Inliers (Loại bỏ nhiễu)
                self.prev_pts = curr_pts[valid][inliers.ravel() == 1].reshape(-1, 1, 2)
            else:
                self.prev_pts = None

        # BƯỚC 3: Cập nhật điểm thích ứng (Adaptive Refresh)
        # Bổ sung điểm đặc trưng mới nếu số điểm hiện tại giảm xuống dưới 40% ngưỡng tối đa
        min_pts_threshold = int(self.max_corners * 0.4)
        if self.prev_pts is None or len(self.prev_pts) < min_pts_threshold:
            self._extract_new_pts(curr_gray, existing_pts=self.prev_pts)

        self.prev_gray = curr_gray
        return H

    def _extract_new_pts(self, gray_img, existing_pts=None):
        """
        Tìm kiếm và bổ sung các điểm đặc trưng Shi-Tomasi mới trên ảnh.

        Args:
            gray_img (numpy.ndarray): Khung hình ảnh xám hiện tại.
            existing_pts (numpy.ndarray, optional): Các điểm đặc trưng đang được theo dõi. Mặc định là None.
        """
        # Tạo mặt nạ vùng quan tâm (ROI mask), loại bỏ 20% khu vực đáy ảnh (vị trí nắp capo xe)
        mask = np.ones_like(gray_img) * 255
        mask[int(gray_img.shape[0]*0.8):, :] = 0 
        
        # Tạo vùng cấm xung quanh các điểm đã tồn tại để tránh trích xuất trùng lặp
        if existing_pts is not None and len(existing_pts) > 0:
            for pt in existing_pts:
                cv2.circle(mask, tuple(pt.ravel().astype(int)), self.min_distance, 0, -1)

        # Tính toán số lượng điểm cần bù đắp thay vì trích xuất toàn bộ
        needed_corners = self.max_corners if existing_pts is None else self.max_corners - len(existing_pts)
        
        if needed_corners > 0:
            new_pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=needed_corners, 
                                            qualityLevel=self.quality_level, 
                                            minDistance=self.min_distance, mask=mask)
            
            # Hợp nhất tập điểm cũ và mới
            if existing_pts is None:
                self.prev_pts = new_pts
            elif new_pts is not None:
                self.prev_pts = np.vstack((existing_pts, new_pts))