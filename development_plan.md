# Kế hoạch Phát triển Hệ thống "Autonomous Traffic Tracker" (10 Tuần)
---

## 🎯 Giai đoạn 1: Nâng cấp Nền tảng Nhận diện (Tuần 1 - 2)
> [!NOTE]
> Mục tiêu: Giải quyết tình trạng Dets < GT Dets (bỏ sót đối tượng quá nhiều).

* **Tuần 1: Cải thiện Detector (YOLO)**
  * Phân tích các frame bị miss (bỏ sót) để xem xét do xe quá nhỏ, quá xa, hay bị che khuất.
  * Thử nghiệm hạ ngưỡng `confidence_threshold` để bắt được các vật thể mờ/nhỏ.
  * Thử nghiệm các phiên bản nâng cao hơn như YOLOv8, YOLOv9, hoặc YOLO11 để so sánh độ chính xác.
* **Tuần 2: Mở rộng Dữ liệu (Data Augmentation & Multi-Dataset)**
  * Không chỉ dùng KITTI, thử nghiệm hệ thống trên các tập dữ liệu phức tạp hơn như **BDD100K** hoặc **Waymo Open Dataset**.
  * Chỉnh sửa `dataloader` để tương thích với các định dạng dữ liệu mới này.

---

## 🎯 Giai đoạn 2: Tối ưu hóa Bộ Trích xuất Đặc trưng (Tuần 3 - 4)
> [!TIP]
> Mục tiêu: Giảm thiểu IDSW (Nhảy ID) do việc nhận diện nhầm các xe có cùng màu hoặc người mặc đồ giống nhau.

* **Tuần 3: Đánh giá mô hình Re-Identification (ReID)**
  * Tách module Extractor, thử nghiệm đo lường độc lập chất lượng của mô hình ReID hiện tại.
  * Thu thập các bức ảnh Crop của những lần bị nhảy ID để phân tích.
* **Tuần 4: Thử nghiệm Extractor mới**
  * Tích hợp các mạng ReID mạnh mẽ hơn (ví dụ: **OSNet**, **BoT-SORT ReID**).
  * Ứng dụng kỹ thuật trích xuất đặc trưng ngoại hình kết hợp với hướng di chuyển (Motion-Aware ReID).

---

## 🎯 Giai đoạn 3: Cuộc cách mạng về Tracking (Tuần 5 - 6)
> [!IMPORTANT]
> Mục tiêu: Nâng cao chỉ số HOTA và IDF1, giảm số lần quỹ đạo bị đứt gãy (Frag) khi đối tượng đi vào vùng che khuất (Occlusion).

* **Tuần 5: Cải tiến Bộ lọc Động học (Unscented Kalman Filter)**
  * Tinh chỉnh (Tuning) các ma trận nhiễu (Q, R) của UKF để dự đoán trơn tru hơn khi xe phanh gấp hoặc chuyển làn nhanh.
  * Cập nhật logic: Vẫn tiếp tục dự đoán vị trí của xe trong 1-2 giây ngay cả khi Detector không nhìn thấy (để chờ đối tượng ló ra khỏi vật cản).
* **Tuần 6: Nâng cấp Thuật toán Kết nối (Association)**
  * Thay vì chỉ kết nối bằng Intersection-over-Union (IoU) truyền thống, hãy tích hợp logic của **ByteTrack** (kết nối 2 chặng: chặng 1 cho box tự tin cao, chặng 2 vét các box mờ) để kéo lại các đối tượng bị bỏ sót.

## 🎯 Điều Kiện Chuyển Giao: Chuẩn Bị Cho Jetson Nano
> [!IMPORTANT]
> Mục tiêu: Trước khi đưa hệ thống xuống chạy trên Jetson Nano (vốn có giới hạn cực lớn về RAM và năng lực tính toán - GPU Maxwell 128 cores), hệ thống phải đạt được các chỉ số sau trên tập Benchmark để đảm bảo độ tin cậy và an toàn:

* **Chỉ số HOTA:** Phải đạt tối thiểu **> 55% - 60%** (Cân bằng giữa phát hiện và bám sát).
* **Chỉ số MOTA:** Phải đạt tối thiểu **> 65% - 70%** (Đủ khả năng không bỏ lọt chướng ngại vật).
* **Chỉ số IDF1:** Phải đạt tối thiểu **> 65% - 70%** (Đảm bảo quỹ đạo nhất quán, hỗ trợ việc phán đoán hướng đi).
* **IDSW & Frag:** Phải giảm xuống tối thiểu **50%** so với hiện tại.

*Lưu ý phần cứng:* Tuyệt đối không dùng mô hình YOLO phiên bản lớn. Bắt buộc tối ưu qua **TensorRT** vì Jetson Nano không hỗ trợ tốt OpenVINO.

---

## 🎯 Giai đoạn 4: Tối ưu hóa Hiệu năng & Triển khai (Tuần 7 - 8)
> [!TIP]
> Mục tiêu: Đẩy FPS từ 37.6 hiện tại lên >60 FPS, sẵn sàng nhúng vào máy tính biên (Jetson Nano) trên xe.

* **Tuần 7: Tăng tốc Cấp thấp (Hardware Acceleration)**
  * Biên dịch các mô hình (YOLO, ReID) sang định dạng **TensorRT** bắt buộc để tương thích với GPU NVIDIA của Jetson Nano.
  * Lựa chọn kiến trúc mô hình cực nhẹ (YOLOv8n / YOLO11n) thay vì bản Medium/Large.
  * Sử dụng kỹ thuật Half-Precision (FP16/INT8) để giảm thiểu độ trễ (Latency).
* **Tuần 8: Xử lý Đa luồng (Multi-threading)**
  * Đưa quá trình đọc video (Dataloader) và hiển thị (Visualizer) sang các luồng độc lập (Asynchronous I/O) để không cản trở pipeline Tracking chính.
  * Tối ưu hóa mã Python bằng Cython hoặc viết lại các logic vòng lặp bằng ma trận NumPy tốc độ cao.

---

## 🎯 Giai đoạn 5: Đánh giá Biên & Triển khai Hệ thống (Tuần 9 - 10)
> [!WARNING]
> Mục tiêu: Đảm bảo tính an toàn cho xe tự hành trong các tình huống thực tế (Edge cases).

* **Tuần 9: Thử nghiệm Tình huống Khắc nghiệt (Edge Cases)**
  * Đưa dữ liệu video quay vào **ban đêm, trời mưa, sương mù** vào để đánh giá độ "lì đòn" của hệ thống.
  * Khắc phục hiện tượng bóng đổ, ánh đèn pha làm sai lệch Bounding Box.
* **Tuần 10: Đóng gói Sản phẩm (Packaging)**
  * Đóng gói toàn bộ hệ thống bằng **Docker**.
  * Chuyển đổi mã nguồn để tích hợp thành một node của **ROS / ROS2** (Robot Operating System), cho phép hệ thống Tracking giao tiếp trực tiếp với hệ thống tự lái của xe.
