"""
=========================================================================================
FILE: test_yolo_fps.py
CHỨC NĂNG: Kiểm thử hiệu năng (Benchmarking) và xuất video kết quả từ mô hình OpenVINO.
-----------------------------------------------------------------------------------------
1. ĐẦU VÀO (INPUT):
    - Weights: Thư mục chứa model OpenVINO (mặc định là bản INT8 đã biên dịch).
    - Dữ liệu: Chuỗi hình ảnh (Sequence) từ tập dữ liệu KITTI/MOT format.
    - Cấu hình: Threshold tự tin (conf_thresh=0.5) và kích thước ảnh đầu vào (imgsz=480).

2. ĐẦU RA (OUTPUT):
    - Video kết quả: 'output_openvino_int8.mp4' đã được vẽ Bounding Box và Label.
    - File định danh (MOT Format): 'det.txt' lưu kết quả detection phục vụ cho bài toán 
      Tracking (Object Tracking) sau này. Định dạng: <frame>, -1, <x>, <y>, <w>, <h>, <conf>, <class>, -1, -1.
    - Log: Chỉ số FPS (Frames Per Second) trung bình thực tế của hệ thống.

3. LOGIC XỬ LÝ ĐẶC TRƯNG:
    - Warm-up: Bỏ qua 10 frame đầu tiên khi tính FPS để loại bỏ độ trễ khởi tạo phần cứng 
      (Hardware latency) và giúp con số thống kê chính xác hơn.
    - Core FPS: Chỉ đo thời gian thực thi hàm `detector.detect()` (Inference time), 
      không tính thời gian đọc file (I/O) hay vẽ ảnh (Drawing) để đánh giá đúng năng lực model.
    - Mapping Label: Chuyển đổi ID lớp của YOLO sang các nhãn cụ thể (Pedestrian, Cyclist, Car).

4. MỤC ĐÍCH HỆ THỐNG:
    - Đánh giá sự đánh đổi giữa độ chính xác (mắt thường xem video) và tốc độ (FPS).
    - Cung cấp dữ liệu đầu vào (det.txt) cho các thuật toán theo dõi như DeepSORT/UKF.
=========================================================================================
"""

import os
import sys
import cv2
import time
import numpy as np

# [Bước 1: Xử lý đường dẫn - Giữ nguyên]
current_file_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.yolo_detector import YOLODetector

def run_fps_test_and_save_video(video_path, model_name="yolo11n_int8_openvino_model", output_filename="output_openvino_int8.mp4"):
    print(f"--- BẮT ĐẦU CHẠY VÀ XUẤT VIDEO CHO {model_name.upper()} ---")
    
    weight_path = os.path.join(project_root, "weights", model_name)
    detector = YOLODetector(model_path=weight_path, conf_thresh=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể đọc được dữ liệu từ {video_path}")
        return

    out_video_path = os.path.join(scripts_dir, output_filename)
    out = None  
    
    frame_count = 0
    total_time = 0.0
    
    mot_labels = {
        0: ("Pedestrian", (0, 165, 255)),  
        1: ("Cyclist", (255, 0, 0)),        
        2: ("Car", (0, 255, 0))
    }

    # ĐÃ LOẠI BỎ: mot_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if out is None:
            h, w = frame.shape[:2] 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(out_video_path, fourcc, 10, (w, h))
            
        frame_count += 1
        
        start_time = time.perf_counter()
        bboxes, confs, class_ids = detector.detect(frame, imgsz=480)
        end_time = time.perf_counter()
        
        process_time = end_time - start_time
        if frame_count > 10:
            total_time += process_time
            
        for bbox, conf, cls_id in zip(bboxes, confs, class_ids):
            x_min, y_min, w, h = bbox
            
            # ĐÃ LOẠI BỎ: Phần tạo mot_line và append vào mot_results

            # Vẽ lên frame (Giữ lại để kiểm tra trực quan)
            x_max, y_max = int(x_min + w), int(y_min + h)
            x_min, y_min = int(x_min), int(y_min)
            
            label_name, color = mot_labels.get(int(cls_id), ("Unknown", (128, 128, 128)))
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            display_text = f"{label_name}: {conf:.2f}"
            cv2.putText(frame, display_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(frame, f"Frame: {frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

        if frame_count % 50 == 0:
            print(f"Đã xử lý xong {frame_count} frames...")

    cap.release()
    if out is not None:
        out.release() 
    
    # ĐÃ LOẠI BỎ: Đoạn code mở file det.txt và ghi dữ liệu (with open...)

    if frame_count > 10:
        valid_frames = frame_count - 10
        avg_fps = valid_frames / total_time
        print(f"\n🚀 FPS Trung bình thực tế: {avg_fps:.2f} FPS")

if __name__ == "__main__":
    # Lưu ý: Sửa lại đường dẫn dataset cho đúng với máy của bạn/đồng đội
    relative_dataset_path = "datasets/KITTI_MOT/KITTI-0000/img1/%06d.jpg" 
    absolute_dataset_path = os.path.join(project_root, relative_dataset_path)
    
    run_fps_test_and_save_video(
        video_path=absolute_dataset_path, 
        model_name="yolo11n_int8_openvino_model", 
        output_filename="output_openvino_int8.mp4"
    )