# Hướng dẫn Huấn luyện YOLO26n trên Google Colab

Tài liệu này hướng dẫn cách sử dụng gói dữ liệu và kịch bản huấn luyện để fine-tune mô hình **YOLO26n** với 6 lớp đối tượng (`person`, `bicycle`, `car`, `motorbike`, `bus`, `truck`) phục vụ cho đề tài NCKH Autonomous Traffic Tracker (triển khai trên NVIDIA Jetson Nano).

---

## 1. Cấu trúc Gói Huấn luyện (`yolo26n_colab_package.zip`)

```text
yolo26n_colab_package/
├── yolo26n_dataset/
│   ├── images/
│   │   ├── train/            # 105 ảnh
│   │   ├── val/              # 30 ảnh
│   │   └── test/             # 15 ảnh
│   ├── labels/
│   │   ├── train/            # 105 file nhãn YOLO
│   │   ├── val/              # 30 file nhãn YOLO
│   │   └── test/             # 15 file nhãn YOLO
│   ├── data.yaml             # File cấu hình đường dẫn và 6 lớp đối tượng
│   └── split_summary.json    # Thống kê chi tiết phân bổ dữ liệu
├── weights/
│   ├── yolo26n_traffic_best.pt # Trọng số fine-tuned 6 lớp để kế thừa huấn luyện tiếp
│   ├── yolo26n.pt              # Trọng số gốc COCO (dùng làm fallback)
│   └── yolo26n.yaml            # Kiến trúc mô hình
├── train_yolo26n_colab.ipynb # Jupyter Notebook chạy từng bước trên Colab
├── train_yolo26n.py          # Script Python huấn luyện bằng dòng lệnh
└── README_COLAB.md           # Hướng dẫn chi tiết này
```

---

## 2. Thông tin Phân chia Dữ liệu (Tỉ lệ 7 : 2 : 1)

* **Tổng số ảnh**: 150 ảnh và 150 nhãn chuẩn đã kiểm duyệt lại từ CVAT (1.470 bounding boxes)
* **Phương pháp phân chia**: Stratified Random (Seed = 42) kết hợp phân bổ đều theo 10 video nguồn (`data01` đến `data10`, mỗi video 15 ảnh) để đảm bảo các cảnh quay khác nhau đều hiện diện ở cả 3 tập, đồng thời giữ đầy đủ 6 lớp đối tượng:
  * **Train (70%)**: 105 ảnh, 1.029 boxes (57 person, 3 bicycle, 283 car, 615 motorbike, 27 bus, 44 truck)
  * **Val (20%)**: 30 ảnh, 304 boxes (11 person, 1 bicycle, 85 car, 173 motorbike, 8 bus, 26 truck)
  * **Test (10%)**: 15 ảnh, 137 boxes (5 person, 1 bicycle, 40 car, 79 motorbike, 5 bus, 7 truck)

---

## 3. Các bước thực hiện trên Google Colab

### Cách 1: Sử dụng Google Colab Notebook (Khuyến nghị - Dễ thao tác nhất)

1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn **File -> Upload notebook** (hoặc **Tải sổ tay lên**) -> Chọn file `train_yolo26n_colab.ipynb`.
3. Bật GPU: Vào **Runtime -> Change runtime type** (Thời lượng -> Thay đổi loại thời lượng) -> Chọn **T4 GPU** -> Nhấn **Save**.
4. Tải dữ liệu lên:
   * Nhấp vào biểu tượng thư mục (Files) ở thanh công cụ bên trái.
   * Kéo thả file `yolo26n_colab_package.zip` vào đó.
5. Chạy tuần tự các ô trong Notebook:
   * **Ô 1 & 2**: Kiểm tra GPU và cài đặt `ultralytics`, `onnx`, `onnxsim`, `onnxruntime`.
   * **Ô 3 & 4**: Tự động giải nén dữ liệu và trực quan hóa ảnh gán nhãn mẫu.
   * **Ô 5**: Bắt đầu fine-tuning YOLO26n với kích thước chữ nhật 16:9 (`imgsz=[384, 640]`, `rect=True`, 60 epochs, batch 16, Early Stopping 20 epochs).
   * **Ô 6 & 7**: Đánh giá mAP trên tập Validation và Test, hiển thị Confusion Matrix và dự đoán ảnh thực tế.
   * **Ô 8**: Xuất mô hình sang ONNX 16:9 (tối ưu với onnxsim) và TensorRT FP16 Engine (kèm lệnh trtexec cho Jetson Nano).
   * **Ô 9**: Tự động đóng gói và tải file `yolo26n_trained_results.zip` về máy tính của bạn.

---

### Cách 2: Sử dụng dòng lệnh qua Terminal trên Colab

Nếu bạn muốn chạy script trực tiếp qua terminal hoặc một ô lệnh:

```bash
# 1. Cài thư viện
!pip install -q "ultralytics>=8.3.0" onnx onnxsim onnxruntime

# 2. Giải nén dữ liệu
!unzip -q yolo26n_colab_package.zip

# 3. Chạy huấn luyện kế thừa từ yolo26n_traffic_best.pt theo chuẩn chữ nhật 16:9 (384x640) & xuất ONNX
!python train_yolo26n.py \
    --data yolo26n_dataset/data.yaml \
    --weights weights/yolo26n_traffic_best.pt \
    --epochs 60 \
    --batch 16 \
    --imgsz 384,640 \
    --rect \
    --patience 20 \
    --export-onnx
```

---

## 4. Tích hợp lại vào Repository Cục bộ & Triển khai Jetson Nano

Sau khi tải `yolo26n_trained_results.zip` từ Colab về máy tính:
1. Giải nén và copy `yolo26n_traffic_best.pt` vào thư mục `weights/`.
2. Copy `yolo26n_traffic_best.onnx` sang máy **NVIDIA Jetson Nano**.
3. Chạy lệnh sau trên Jetson Nano để nén sang **TensorRT Engine (FP16)** cực đại tốc độ:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
       --onnx=yolo26n_traffic_best.onnx \
       --saveEngine=weights/yolo26n_traffic_fp16.engine \
       --fp16
   ```
4. Mô hình mới đã sẵn sàng chạy với pipeline C++ / Python TensorRT trên Jetson Nano!

