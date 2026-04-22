"""
Mô đun cung cấp công cụ phân tích cú pháp (Parser) cho bộ dữ liệu KITTI MOT.

Mô đun này chịu trách nhiệm nạp hình ảnh và đọc thông tin nhận diện (detections/ground truth) 
từ thư mục theo chuẩn cấu trúc MOT Challenge. Nó đóng vai trò là nguồn cung cấp dữ liệu 
đầu vào tuần tự (frame-by-frame) cho hệ thống Tracking.
"""

import os
import numpy as np
import cv2

class KittiParser:
    """
    Trình phân tích cú pháp (Parser) đọc dữ liệu từ bộ dữ liệu KITTI chuẩn MOT.

    Lớp này hỗ trợ việc tải các tệp tin ảnh và trích xuất mảng tọa độ hộp bao 
    [x, y, w, h, conf] tương ứng với từng khung hình.

    Attributes:
        seq_dir (str): Đường dẫn tới thư mục chứa chuỗi video (sequence).
        img_dir (str): Đường dẫn tới thư mục con 'img1' chứa các tệp hình ảnh.
        img_files (list): Danh sách tên các tệp hình ảnh hợp lệ đã được sắp xếp.
        raw_detections (numpy.ndarray): Mảng 2D chứa dữ liệu nhận diện thô từ tệp văn bản.
    """

    def __init__(self, seq_dir):
        """
        Khởi tạo KittiParser cho một chuỗi video cụ thể.

        Kiểm tra và thiết lập danh sách ảnh đầu vào. Đọc tệp 'det.txt' hoặc 'gt.txt' 
        để tải trước toàn bộ dữ liệu nhận diện lên bộ nhớ.

        Args:
            seq_dir (str): Đường dẫn tới thư mục chứa dữ liệu chuỗi video.

        Raises:
            FileNotFoundError: Nếu không tìm thấy thư mục ảnh 'img1'.
            ValueError: Nếu thư mục 'img1' trống hoặc không chứa tệp ảnh hợp lệ.
        """
        self.seq_dir = seq_dir
        self.img_dir = os.path.join(seq_dir, 'img1')

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Không tìm thấy: {self.img_dir}")
        
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_exts)])

        if len(self.img_files) == 0:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ trong: {self.img_dir}")
        
        det_path = os.path.join(seq_dir, 'det', 'det.txt')
        
        if not os.path.exists(det_path):
            det_path = os.path.join(seq_dir, 'gt', 'gt.txt')
        
        if os.path.exists(det_path):
            self.raw_detections = np.loadtxt(det_path, delimiter=',')
            print(f"[INFO] Đã tải {len(self.img_files)} ảnh và {len(self.raw_detections)} hộp bao từ chuỗi {os.path.basename(seq_dir)}.")
        else:
            self.raw_detections = np.empty((0, 10))
            print(f"[WARNING] Không tìm thấy tệp det.txt hoặc gt.txt trong {seq_dir}. Khởi tạo mảng nhận diện rỗng.")

    def get_frame(self):
        """
        Khởi tạo một generator để cung cấp từng khung hình và dữ liệu nhận diện tương ứng.

        Duyệt qua danh sách ảnh, tải ảnh bằng OpenCV và trích xuất các hộp bao thuộc về 
        khung hình đó từ mảng `raw_detections`.

        Yields:
            tuple: Chứa 3 phần tử:
                - frame_idx (int): Chỉ số của khung hình hiện tại.
                - image (numpy.ndarray): Ma trận ảnh BGR kích thước (H, W, 3).
                - out_dets (numpy.ndarray): Mảng 2D kích thước (N, 6) chứa tọa độ và thông tin 
                  [x, y, w, h, conf, class_id]. Trả về mảng rỗng nếu không có đối tượng.
        """
        for img_name in self.img_files:
            frame_idx = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(self.img_dir, img_name)

            image = cv2.imread(img_path)

            if image is None:
                continue

            # Lọc các bounding box thuộc về khung hình hiện tại (cột 0 là frame_idx)
            mask = self.raw_detections[:, 0] == frame_idx
            frame_dets = self.raw_detections[mask]

            if len(frame_dets) > 0:
                # Lấy các cột tương ứng: [x, y, w, h, conf, class_id]
                out_dets = frame_dets[:, [2, 3, 4, 5, 6, 7]].astype(np.float32)
            else:
                out_dets = np.empty((0, 6), dtype=np.float32)

            yield frame_idx, image, out_dets
        
if __name__ == "__main__":
    test_dir = "datasets/KITTI_MOT/KITTI-0000"

    if os.path.exists(test_dir):
        parser = KittiParser(test_dir)
        for frame_idx, img, dets in parser.get_frame():
            print(f"[INFO] Khung hình {frame_idx:04d} | Kích thước ảnh: {img.shape} | Số lượng đối tượng: {len(dets)}")
            # Dừng sau khung hình đầu tiên để kiểm thử
            break
    else:
        print(f"[ERROR] Không tìm thấy thư mục kiểm thử: {test_dir}. Vui lòng kiểm tra lại đường dẫn.")