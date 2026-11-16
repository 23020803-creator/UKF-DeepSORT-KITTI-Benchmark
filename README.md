# Autonomous Traffic Tracker (NVIDIA Jetson Nano)

Hệ thống theo dõi giao thông đa đối tượng (Multi-Object Traffic Tracking) thời gian thực, thiết kế chuyên biệt cho máy tính nhúng **NVIDIA Jetson Nano** (GPU Maxwell 128 lõi, JetPack) và các hệ thống xe tự hành / camera giám sát giao thông đô thị.

Hệ thống hỗ trợ đầy đủ **6 phân lớp đối tượng giao thông thực tế**:
1. **Person** (Người đi bộ)
2. **Bicycle** (Xe đạp)
3. **Car** (Xe ô tô con)
4. **Motorbike** (Xe máy)
5. **Bus** (Xe buýt)
6. **Truck** (Xe tải)

---

## 🌟 Kiến trúc & Tính năng chính

- **Phát hiện đối tượng (Detection)**: Mô hình YOLO26n được fine-tune trên dữ liệu giao thông hỗn hợp, đạt **76.04% mAP@0.5** và **66.35% mAP@0.5:0.95**.
- **Định danh ngoại hình (ReID)**: Sử dụng mạng MobileNetV4 Unified ReID nhẹ (256-D embedding), phục vụ nhận diện cả người và phương tiện.
- **Bù trừ chuyển động camera (CMC)**: Tích hợp Sparse Optical Flow để xử lý rung lắc và chuyển động của camera trên phương tiện di chuyển.
- **Dự đoán quỹ đạo phi tuyến**: Sử dụng **Unscented Kalman Filter (UKF)** thay thế KF tuyến tính, mô hình hóa chính xác động học quay đầu và tăng giảm tốc của phương tiện giao thông.
- **Mục tiêu triển khai chính (Target: Jetson Nano)**: Toàn bộ pipeline được tối ưu bằng **C++ / NVIDIA TensorRT FP16** (trong thư mục `cpp/`), đảm bảo tốc độ xử lý thời gian thực trên phần cứng nhúng.
- **Môi trường Laptop (Prototyping & Testing)**: Chạy thử nghiệm nhanh qua Python và PyTorch `.pt` để kiểm thử thuật toán và trực quan hóa kết quả.

---

## 📂 Cấu trúc dự án

- `cpp/`: **Mã nguồn triển khai thực tế trên Jetson Nano** (C++, CUDA, TensorRT Engine FP16, OpenCV). Xem [hướng dẫn chi tiết Jetson](cpp/README.md).
- `src/`: Lõi thuật toán phiên bản Python để thử nghiệm nhanh (Detector, Tracker, UKF, CMC, ReID, Visualizer).
- `weights/`: Trọng số mô hình:
  - `yolo26n_traffic_best.pt`: Mô hình YOLO26n đã huấn luyện 6 lớp giao thông.
  - `yolo26n.pt`: Mô hình YOLO gốc (COCO).
  - `unified/unified_reid.onnx`: Trọng số trích xuất đặc trưng ReID dùng để build TensorRT Engine.
- `datasets/`: Tập dữ liệu kiểm thử và cấu hình dữ liệu (`yolo26n_dataset/`).
- `tools/`: Các công cụ hỗ trợ:
  - `export_tensorrt.py`: Xuất mô hình sang TensorRT FP16 cho Jetson Nano.
  - `package_cvat_yolo.py`: Đóng gói nhãn chuẩn CVAT YOLO.
  - `auto_label.py`: Tự động gán nhãn hỗ trợ.
- `test_yolo_accuracy.py`: Kịch bản đánh giá & so sánh đối đầu chi tiết giữa mô hình gốc và mô hình đã huấn luyện.
- `outputs/`: Báo cáo đánh giá JSON, ảnh dự đoán so sánh, và video kết quả tracking.

---

## 🛠️ Triển khai trên NVIDIA Jetson Nano (C++ / TensorRT)

Xem hướng dẫn chi tiết tại [`cpp/README.md`](cpp/README.md):

1. **Biên dịch mô hình sang TensorRT FP16**:
   ```bash
   # Cách 1: Xuất trực tiếp qua Ultralytics trên Jetson
   python tools/export_tensorrt.py --weights weights/yolo26n_traffic_best.pt

   # Cách 2: Sử dụng trtexec của JetPack
   /usr/src/tensorrt/bin/trtexec --onnx=weights/unified/unified_reid.onnx --saveEngine=weights/unified/unified_reid_fp16.engine --fp16
   ```

2. **Biên dịch mã nguồn C++**:
   ```bash
   cd cpp
   chmod +x build.sh
   ./build.sh
   ```

3. **Chạy hệ thống theo dõi trực tiếp**:
   ```bash
   # Chạy với file video test hoặc webcam/camera CSI (0)
   ./build/ukf_deepsort_app weights/yolo26n_fp16.engine weights/unified/unified_reid_fp16.engine input_traffic.mp4
   ```

---

## 💻 Chạy thử nghiệm nhanh trên Laptop (Python)

Trên laptop, hệ thống sử dụng PyTorch `.pt` để chạy thử nghiệm tiện lợi:

1. **Cài đặt thư viện**:
   ```bash
   pip install ultralytics torch torchvision opencv-python scipy filterpy pyyaml
   ```

2. **So sánh độ chính xác phát hiện (YOLO Base vs Fine-tuned)**:
   ```bash
   python test_yolo_accuracy.py --conf 0.5
   ```
   Kết quả đánh giá 6 lớp đối tượng và ảnh so sánh trực quan được lưu tại `outputs/evaluation/yolo26n_benchmark/`.

3. **Chạy luồng tracking Python**:
   ```bash
   python src/main.py
   ```
