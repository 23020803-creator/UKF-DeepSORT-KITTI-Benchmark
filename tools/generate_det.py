"""
Kịch bản trích xuất Bounding Box từ YOLO và lưu thành tệp det.txt theo chuẩn MOT.

Công cụ này cho phép chạy phát hiện đối tượng ngoại tuyến (offline detections) 
trên các chuỗi video KITTI và ghi kết quả vào thư mục `det/det.txt` của từng sequence.
Định dạng MOT16: <frame>, -1, <x>, <y>, <w>, <h>, <conf>, <class_id>, -1, -1
"""

import os
import sys
import time
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataloader.dataloader import KittiParser
from src.detector.detector import YOLODetector

def generate_detections_for_sequence(seq_name, data_dir="datasets/KITTI_MOT", model_path="weights/yolo26n_int8_openvino_model", conf_thresh=0.5):
    """
    Sinh file det/det.txt cho một chuỗi video đơn lẻ.
    """
    seq_dir = os.path.join(data_dir, seq_name)
    if not os.path.exists(seq_dir):
        print(f"[ERROR] Không tìm thấy thư mục sequence: {seq_dir}")
        return

    det_dir = os.path.join(seq_dir, "det")
    os.makedirs(det_dir, exist_ok=True)
    det_file_path = os.path.join(det_dir, "det.txt")

    parser = KittiParser(seq_dir)
    detector = YOLODetector(model_path=model_path, conf_thresh=conf_thresh)

    lines = []
    print(f"[INFO] Đang tạo detections cho {seq_name}...")
    start_time = time.time()
    frame_count = 0

    for frame_idx, image, _ in parser.get_frame():
        frame_count += 1
        bboxes, confs, class_ids = detector.detect(image)
        for bbox, conf, cls_id in zip(bboxes, confs, class_ids):
            x, y, w, h = bbox
            line = f"{frame_idx},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{int(cls_id)},-1,-1\n"
            lines.append(line)

    with open(det_file_path, "w") as f:
        f.writelines(lines)

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"[SUCCESS] {seq_name}: Đã ghi {len(lines)} detections vào {det_file_path} ({fps:.1f} FPS)")

def main():
    data_dir = "datasets/KITTI_MOT"
    if not os.path.exists(data_dir):
        print(f"[ERROR] Không tìm thấy: {data_dir}")
        return

    seqs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("KITTI-")])
    print(f"[INFO] Tìm thấy {len(seqs)} sequences để sinh det.txt")

    for seq in seqs:
        generate_detections_for_sequence(seq, data_dir=data_dir)

if __name__ == "__main__":
    main()