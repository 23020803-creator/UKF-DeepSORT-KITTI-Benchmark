# Class đọc ảnh và file det.txt liên tục cho main_kitti.py.
import os
import numpy as np
import cv2

class KittiParser:
    """ Class đọc dữ liệu chuẩn MOT format từ KITTI dataset. 
    Cung cấp frame ảnh và mảng tọa độ [x, y, w, h, conf] cho hệ thống
    """
    def __init__(self, seq_dir):
        self.seq_dir = seq_dir
        self.img_dir = os.path.join(seq_dir, 'img1')

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Không tìm thấy: {self.img_dir}")
        
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_exts)])

        if len(self.img_files) == 0:
            raise ValueError(f"Không tìm thấy ảnh trong: {self.img_dir}")
        
        det_path = os.path.join(seq_dir, 'det', 'det.txt')
        
        if not os.path.exists(det_path):
            det_path = os.path.join(seq_dir, 'gt', 'gt.txt')
        
        if os.path.exists(det_path):
            self.raw_detections = np.loadtxt(det_path, delimiter=',')
            print(f"[+] Đã load {len(self.img_files)} ảnh và {len(self.raw_detections)} boxes từ {os.path.basename(seq_dir)}.")
        else:
            self.raw_detections = np.empty((0, 10))
            print(f"[!] Cảnh báo: Không tìm thấy file det.txt hoặc gt.txt trong {seq_dir}")

    def get_frame(self):
            """
            Generator yield từng frame ảnh.
            Output: frame_idx(int), image(H,W,3), detections(N,5) [x, y, w, h, conf]
            """
            for img_name in self.img_files:
                frame_idx = int(os.path.splitext(img_name)[0])
                img_path = os.path.join(self.img_dir, img_name)

                image = cv2.imread(img_path)

                if image is None:
                    continue

                mask = self.raw_detections[:, 0] == frame_idx
                frame_dets = self.raw_detections[mask]

                if len(frame_dets) > 0:
                    out_dets = frame_dets[:, [2, 3, 4, 5, 6]].astype(np.float32)
                else:
                    out_dets = np.empty((0, 5), dtype=np.float32)

                yield frame_idx, image, out_dets
        
if __name__ == "__main__":
    test_dir = "datasets/KITTI_MOT/KITTI-0000"

    if os.path.exists(test_dir):
        parser = KittiParser(test_dir)
        for frame_idx, img, dets in parser.get_frame():
            print(f"Frame {frame_idx:04d} | Kích thước ảnh: {img.shape} | Số lượng xe: {len(dets)}")

            break
    else:
        print(f"Không tìm thấy thư mục: {test_dir}. Vui lòng kiểm tra đường dẫn.")
