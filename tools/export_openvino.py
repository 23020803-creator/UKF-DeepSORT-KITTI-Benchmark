"""
Kịch bản lượng tử hóa và biên dịch mô hình YOLO sang chuẩn OpenVINO INT8.

Mô đun này đảm nhiệm việc tải mô hình PyTorch (yolo11n.pt) và xuất khẩu 
(export) sang định dạng OpenVINO Intermediate Representation (IR). 
Điểm cốt lõi là quá trình lượng tử hóa sau huấn luyện (Post-Training Quantization - PTQ) 
về chuẩn 8-bit Integer (INT8).

Việc lượng tử hóa INT8 kết hợp cùng OpenVINO Toolkit giúp thu nhỏ kích thước 
mô hình (~4 lần) và tăng tốc độ suy luận (FPS) một cách đột phá trên các nền 
tảng phần cứng Intel (CPU), cực kỳ phù hợp cho môi trường Edge AI. Quá trình 
hiệu chuẩn (Calibration) sử dụng tập dữ liệu `coco8.yaml` để giảm thiểu suy hao 
độ chính xác (Accuracy Drop).
"""

import os
import sys
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
weights_dir = os.path.join(project_root, "weights")

def export_model_int8():
    """
    Thực thi quá trình lượng tử hóa và xuất mô hình OpenVINO.

    Hàm này thực hiện các công việc sau:
    1. Định vị mô hình PyTorch gốc (`yolo11n.pt`) trong thư mục weights.
    2. Khởi tạo đối tượng YOLO từ mô hình.
    3. Kích hoạt tính năng export của Ultralytics với các tham số:
       - format="openvino": Xuất ra định dạng IR (.xml và .bin).
       - int8=True: Kích hoạt bộ lượng tử hóa PTQ.
       - data="coco8.yaml": Cấp tập dữ liệu mẫu để tính toán Scale và Zero-point.
       - imgsz=480: Ghim cứng kích thước đầu vào (Static Shape).
       - task='detect': Chỉ định bài toán là phát hiện vật thể.
    """
    print("[INFO] BẮT ĐẦU BIÊN DỊCH MODEL OPENVINO (PHIÊN BẢN INT8 TỐI ƯU FPS)")
    
    pt_model_path = os.path.join(weights_dir, "yolo11n.pt")
    
    if not os.path.exists(pt_model_path):
        print(f"[ERROR] Không tìm thấy mô hình PyTorch gốc tại: {pt_model_path}")
        print("[HINT] Vui lòng tải file yolo11n.pt vào thư mục weights trước khi chạy.")
        return

    print(f"[INFO] Tải mô hình gốc: {pt_model_path}")
    model = YOLO(pt_model_path)
    
    print("[INFO] Đang tiến hành lượng tử hóa (Quantization) và hiệu chuẩn (Calibration)...")
    # Kích hoạt ép kiểu INT8 và cấp tập dữ liệu để Calibration
    model.export(
        format="openvino", 
        int8=True,          # Bật lượng tử hóa 8-bit
        data="coco8.yaml",  # Dữ liệu mẫu để đo lường Scale/Zero-point (Ultralytics tự tải nếu thiếu)
        imgsz=480,          
        task='detect'       
    )
    
    print("[SUCCESS] Lượng tử hóa INT8 và biên dịch OpenVINO hoàn tất!")

if __name__ == "__main__":
    export_model_int8()