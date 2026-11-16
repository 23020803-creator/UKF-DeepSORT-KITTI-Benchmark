# UKF-DeepSORT C++ Deployment for NVIDIA Jetson Nano

Dự án triển khai thuật toán **Multi-Object Tracking (UKF-DeepSORT)** bằng **C++17** tối ưu hóa đặc biệt cho thiết bị nhúng **NVIDIA Jetson Nano** (Tegra X1, 128 Maxwell GPU cores, ARM Cortex-A57 CPU).

---

## 1. Yêu Cầu Hệ Thống & Cài Đặt Trên Jetson Nano

* **Phần cứng**: NVIDIA Jetson Nano (4GB Developer Kit).
* **Hệ điều hành**: JetPack 4.6+ (Ubuntu 18.04 LTS aarch64) hoặc JetPack 5.x.
* **Thư viện phụ thuộc**:
  * CUDA Toolkit (đã có sẵn trong JetPack).
  * TensorRT 8.x (đã có sẵn trong JetPack).
  * OpenCV 4.x (`libopencv-dev`).
  * Eigen 3.3+ (`libeigen3-dev`).

### Cài đặt nhanh thư viện trên Jetson Nano:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libeigen3-dev libopencv-dev
```

---

## 2. Bước 1: Biên Dịch TensorRT Engine (Chạy trên Jetson Nano)

Từ thư mục gốc dự án, bạn có 2 cách xuất Engine TensorRT FP16:

### Cách 1: Sử dụng lệnh yolo export (Khuyến nghị, nhanh nhất)
```bash
yolo export model=weights/yolo26n_traffic_best.pt format=engine half=True imgsz=640 device=0
# Lệnh trên sẽ sinh ra file weights/yolo26n_traffic_best.engine
```

### Cách 2: Sử dụng công cụ `trtexec` của JetPack
```bash
# 1. Biên dịch YOLO Detection (6 classes Traffic fine-tuned)
/usr/src/tensorrt/bin/trtexec \
    --onnx=weights/yolo26n_traffic_best.onnx \
    --saveEngine=weights/yolo26n_fp16.engine \
    --fp16

# 2. Biên dịch Unified ReID (MobileNetV4 128x128 256D)
/usr/src/tensorrt/bin/trtexec \
    --onnx=weights/unified/unified_reid.onnx \
    --saveEngine=weights/unified/unified_reid_fp16.engine \
    --fp16
```

---

## 3. Bước 2: Biên Dịch Mã Nguồn C++

Chỉ cần chạy script tự động:
```bash
cd cpp
chmod +x build.sh
./build.sh
```

Hoặc biên dịch thủ công qua CMake:
```bash
cd cpp
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

---

## 4. Bước 3: Chạy Thực Nghiệm

```bash
cd cpp/build

# Chạy với Webcam USB / Camera CSI
./ukf_deepsort 0 ../../weights/yolo26n_fp16.engine ../../weights/unified/unified_reid_fp16.engine

# Chạy với file Video test
./ukf_deepsort path/to/video.mp4 ../../weights/yolo26n_fp16.engine ../../weights/unified/unified_reid_fp16.engine
```

---

## 5. Cấu Trúc Mã Nguồn

* `include/core/UKF.hpp` & `src/core/UKF.cpp`: Bộ lọc Unscented Kalman Filter 8D (CTRV + gamma expansion) sử dụng Eigen 3.
* `include/core/CMC.hpp` & `src/core/CMC.cpp`: Bù trừ chuyển động camera (Lucas-Kanade + RANSAC Affine).
* `include/core/Matching.hpp` & `src/core/Matching.cpp`: Khoảng cách Cosine, IoU/IoM, Mahalanobis (Cholesky), Hungarian.
* `include/core/Track.hpp` & `src/core/Track.cpp`: Quản lý trạng thái và vector đặc trưng ngoại hình EMA.
* `include/core/Tracker.hpp` & `src/core/Tracker.cpp`: Bộ điều phối 3 vòng ghép cặp cascade.
* `include/models/TrtEngine.hpp`: Wrapper quản lý nạp và suy luận TensorRT GPU.
* `include/models/YoloDetector.hpp`: Nhận diện vật thể YOLO và NMS C++.
* `include/models/ReidExtractor.hpp`: Trích xuất đặc trưng ReID Unified 256 chiều.
