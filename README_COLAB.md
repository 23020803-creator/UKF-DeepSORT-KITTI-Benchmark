# Hướng dẫn Huấn luyện YOLO26n trên Google Colab

Tài liệu này hướng dẫn cách sử dụng gói dữ liệu và kịch bản huấn luyện để fine-tune mô hình **YOLO26n** với 6 lớp đối tượng (`person`, `bicycle`, `car`, `motorbike`, `bus`, `truck`) phục vụ cho đề tài NCKH Autonomous Traffic Tracker (triển khai trên NVIDIA Jetson Nano).

---

## 1. Cấu trúc Gói Huấn luyện (`yolo26n_colab_package.zip`)

```text
yolo26n_colab_package/
├── yolo26n_dataset/
│   ├── images/
│   │   ├── train/            # 70 ảnh
│   │   ├── val/              # 20 ảnh
│   │   └── test/             # 10 ảnh
│   ├── labels/
│   │   ├── train/            # 70 file nhãn YOLO
│   │   ├── val/              # 20 file nhãn YOLO
│   │   └── test/             # 10 file nhãn YOLO
│   ├── data.yaml             # File cấu hình đường dẫn và 6 lớp đối tượng
│   └── split_summary.json    # Thống kê chi tiết phân bổ dữ liệu
├── weights/
│   ├── yolo26n.pt            # Trọng số tiền huấn luyện YOLO26n
│   └── yolo26n.yaml          # Kiến trúc mô hình
├── train_yolo26n_colab.ipynb # Jupyter Notebook chạy từng bước trên Colab
├── train_yolo26n.py          # Script Python huấn luyện bằng dòng lệnh
└── README_COLAB.md           # Hướng dẫn chi tiết này
```

---

## 2. Thông tin Phân chia Dữ liệu (Tỉ lệ 7 : 2 : 1)

* **Tổng số ảnh**: 100 ảnh và 100 nhãn tương ứng (941 bounding boxes)
* **Phương pháp phân chia**: Stratified Random (Seed = 1) kết hợp phân bổ đều theo video nguồn (`data01` đến `data07`) để đảm bảo các cảnh quay khác nhau đều hiện diện ở cả 3 tập:
  * **Train (70%)**: 70 ảnh, 651 boxes (có đầy đủ 6 lớp: 45 person, 1 bicycle, 215 car, 340 motorbike, 23 bus, 27 truck)
  * **Val (20%)**: 20 ảnh, 193 boxes (có đầy đủ 6 lớp: 19 person, 1 bicycle, 64 car, 96 motorbike, 6 bus, 7 truck)
  * **Test (10%)**: 10 ảnh, 97 boxes (đánh giá kiểm thử độc lập: 6 person, 27 car, 56 motorbike, 6 bus, 2 truck)

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
   * **Ô 1 & 2**: Kiểm tra GPU và cài đặt `ultralytics`, `openvino`, `onnx`.
   * **Ô 3 & 4**: Tự động giải nén dữ liệu và trực quan hóa ảnh gán nhãn mẫu.
   * **Ô 5**: Bắt đầu fine-tuning YOLO26n (mặc định 60 epochs, batch 16, imgsz 640, Early Stopping 20 epochs).
   * **Ô 6 & 7**: Đánh giá mAP trên tập Validation và Test, hiển thị Confusion Matrix và dự đoán ảnh thực tế.
   * **Ô 8**: Xuất mô hình sang ONNX, OpenVINO FP16 và OpenVINO INT8 (PTQ tối ưu tốc độ CPU).
   * **Ô 9**: Tự động tải file `yolo26n_trained_results.zip` về máy tính của bạn.

---

### Cách 2: Sử dụng dòng lệnh qua Terminal trên Colab

Nếu bạn muốn chạy script trực tiếp qua terminal hoặc một ô lệnh:

```bash
# 1. Cài thư viện
!pip install -q "ultralytics>=8.3.0" openvino nncf onnx

# 2. Giải nén dữ liệu
!unzip -q yolo26n_colab_package.zip

# 3. Chạy huấn luyện
!python train_yolo26n.py \
    --data yolo26n_dataset/data.yaml \
    --weights weights/yolo26n.pt \
    --epochs 60 \
    --batch 16 \
    --imgsz 640 \
    --patience 20 \
    --export-openvino
```

---

## 4. Tích hợp lại vào Repository Cục bộ

Sau khi tải `yolo26n_trained_results.zip` từ Colab về máy tính:
1. Giải nén và copy `yolo26n_best.pt` vào thư mục `weights/`.
2. Copy thư mục OpenVINO (`yolo26n_openvino_model/` hoặc `yolo26n_int8_openvino_model/`) vào thư mục `weights/`.
3. Mô hình mới đã sẵn sàng cho pipeline Autonomous Traffic Tracker trên Jetson Nano!

