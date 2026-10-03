# Autonomous-Traffic-Tracker

Hệ thống theo dõi giao thông đa đối tượng (Multi-object tracking) chuyên dụng cho xe tự hành, tập trung vào việc theo dõi ô tô và người đi bộ. Dự án cung cấp một bộ khung linh hoạt cho phép tích hợp các thuật toán hiện đại (như YOLO, ReID) kết hợp với các bộ lọc tiên tiến (như Unscented Kalman Filter - UKF) và bù trừ chuyển động camera (CMC) để đảm bảo độ bám sát chính xác và mượt mà trong điều kiện thực tế.

## 🌟 Tính năng chính

- **Phát hiện đối tượng**: Tích hợp YOLO (hỗ trợ các phiên bản YOLO mới nhất) được tối ưu hóa bằng OpenVINO INT8.
- **Trích xuất đặc trưng (ReID)**: Sử dụng các mô hình Unified ReID / OSNet để nhận diện lại đối tượng ổn định.
- **Bù trừ chuyển động (CMC)**: Tích hợp Sparse Optical Flow để xử lý nhiễu do chuyển động của camera trên xe tự lái.
- **Dự đoán quỹ đạo**: Thay thế Kalman Filter tuyến tính truyền thống bằng **Unscented Kalman Filter (UKF)** để xử lý tốt hơn các chuyển động phi tuyến tính phức tạp trong thực tế.
- **Tối ưu phần cứng**: Hỗ trợ tăng tốc hiệu năng chạy suy luận (inference) trên CPU/iGPU với hệ sinh thái OpenVINO và NNCF.

## 📂 Cấu trúc thư mục

- `core/`: Lõi thuật toán (Tracker, Unscented Kalman Filter, gán ghép cặp Hungarian, CMC) và các module mô hình AI (YOLO Detector, ReID Extractor).
- `datasets/`: Trình phân tích tập dữ liệu KITTI (KittiParser).
- `evaluation/` & `tools/`: Các tập lệnh hỗ trợ xuất mô hình sang OpenVINO, lượng tử hóa và benchmark đánh giá chất lượng (TrackEval).
- `weights/`: Nơi lưu trữ trọng số (weights) của các mô hình đã được huấn luyện hoặc lượng tử hóa.

## 🛠️ Yêu cầu hệ thống

Dự án yêu cầu Python 3.8+ và các thư viện chính sau:
- `ultralytics` (Phát hiện đối tượng)
- `openvino>=2024.0.0`, `nncf>=2.9.0` (Tối ưu hóa mô hình AI)
- `torch`, `torchvision` (Framework Deep Learning)
- `filterpy` (Cung cấp các bộ lọc UKF)
- `opencv-python`, `scipy`, `numpy` (Xử lý dữ liệu và toán học)

## 🚀 Hướng dẫn cài đặt

1. **Clone dự án**:
   ```bash
   git clone https://github.com/23020803-creator/UKF-DeepSORT-KITTI-Benchmark.git
   cd UKF-DeepSORT-KITTI-Benchmark
   ```

2. **Cài đặt các thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị Dữ liệu & Weights**:
   - Tải bộ dữ liệu hình ảnh KITTI và đưa vào cấu trúc `datasets/KITTI_MOT/`.
   - Đảm bảo các mô hình học sâu cần thiết (YOLO, ReID) đã nằm sẵn trong thư mục `weights/`.

## 🎯 Chạy kiểm thử hệ thống

Khởi chạy script `src/main.py` để test luồng tracking hoàn chỉnh (CMC -> YOLO -> ReID -> UKF -> Visualize) trên một sequence KITTI:

```bash
python src/main.py
```

*Video kết quả sau khi phân tích sẽ được xuất và lưu tự động vào thư mục `outputs/videos/`.*
