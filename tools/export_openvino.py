"""
=========================================================================================
FILE: export_openvino.py
CHỨC NĂNG: Chuyển đổi và tối ưu hóa mô hình YOLO11 từ PyTorch (.pt) sang OpenVINO INT8.
-----------------------------------------------------------------------------------------
1. ĐẦU VÀO (INPUT):
    - Model gốc: 'yolo11n.pt' (định dạng Floating Point 32 - FP32).
    - Calibration Data: 'coco8.yaml' (Tập dữ liệu nhỏ gồm 8 ảnh để thực hiện Calibration).

2. ĐẦU RA (OUTPUT):
    - Thư mục 'yolo11n_openvino_model/' chứa:
        + metadata.yaml: Thông tin cấu hình mô hình.
        + *.xml: Kiến trúc mạng (Intermediate Representation).
        + *.bin: Trọng số đã được lượng tử hóa sang INT8.

3. TẠI SAO DÙNG INT8 & OPENVINO?
    - Chế độ INT8 (8-bit Integer) giúp giảm dung lượng model (~4 lần so với FP32) và tăng 
      tốc độ xử lý (FPS) đáng kể trên các phần cứng của Intel (CPU, iGPU, NPU).
    - Quá trình chuyển đổi sử dụng Post-Training Quantization (PTQ): Sử dụng tập dữ liệu 
      'coco8.yaml' để tính toán các tham số Scale và Zero-point, đảm bảo giảm độ chính xác 
      (Accuracy Drop) ở mức thấp nhất trong khi nén dữ liệu.

4. LƯU Ý KỸ THUẬT:
    - imgsz=480: Ảnh đầu vào sẽ được fix ở kích thước 480x480.
    - Phù hợp triển khai trên các hệ thống nhúng hoặc PC không có GPU rời mạnh.
=========================================================================================
"""

import os
import sys
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
weights_dir = os.path.join(project_root, "weights")

def export_model_int8():
    print("--- BẮT ĐẦU BIÊN DỊCH MODEL OPENVINO (BẢN INT8 TỐI ƯU FPS) ---")
    
    pt_model_path = os.path.join(weights_dir, "yolo11n.pt")
    model = YOLO(pt_model_path)
    
    # Kích hoạt ép kiểu INT8 và cấp tập dữ liệu để Calibration
    model.export(
        format="openvino", 
        int8=True,          # <--- Bật lượng tử hóa 8-bit
        data="coco8.yaml",  # <--- Dữ liệu mẫu để đo lường Scale/Zero-point (Ultralytics tự động tải nếu chưa có)
        imgsz=480,          
        task='detect'       
    )
    
    print("\n✅ Lượng tử hóa INT8 và biên dịch hoàn tất!")

if __name__ == "__main__":
    export_model_int8()