"""
Mô đun cung cấp công cụ trực quan hóa (Visualizer) cho hệ thống Tracking.

Mô đun này chứa lớp Visualizer giúp vẽ các hộp bao dự đoán (bounding boxes), thông tin 
định danh (ID), tên phân lớp đối tượng (class name), đồng thời hỗ trợ ghi chuỗi 
khung hình kết quả thành tệp video hoàn chỉnh (.mp4).
"""

import os
import cv2
import numpy as np

class Visualizer:
    """
    Lớp hỗ trợ trực quan hóa kết quả theo dõi đa đối tượng và lưu xuất video.

    Attributes:
        output_path (str): Đường dẫn lưu tệp video kết quả đầu ra.
        fps (int): Tốc độ khung hình (Frames Per Second) của video đầu ra.
        writer (cv2.VideoWriter): Đối tượng ghi video của thư viện OpenCV.
        colors_palette (numpy.ndarray): Mảng chứa 1000 mã màu ngẫu nhiên nhưng cố định 
            được cấp phát trước cho các ID nhằm đảm bảo tính nhất quán.
        class_names (dict): Từ điển ánh xạ từ Class ID sang chuỗi tên đối tượng.
    """
    
    def __init__(self, output_path="outputs/videos/results.mp4", fps=30):
        """
        Khởi tạo đối tượng Visualizer.

        Tạo trước một bảng màu chuẩn ngẫu nhiên từ hạt giống (seed) cố định để 
        đảm bảo màu sắc của các ID không bị thay đổi giữa các lần chạy.

        Args:
            output_path (str, optional): Đường dẫn tệp video kết quả. Mặc định "outputs/videos/results.mp4".
            fps (int, optional): Tốc độ ghi khung hình cho video xuất ra. Mặc định là 30.
        """
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        
        # Tạo bảng màu cố định ngẫu nhiên cho 1000 ID đầu tiên
        np.random.seed(42)
        self.colors_palette = np.random.randint(0, 255, size=(1000, 3), dtype=int)
        
        # Từ điển ánh xạ từ Class ID sang String (6 lớp đối tượng giao thông thực tế)
        self.class_names = {
            0: "Person",
            1: "Bicycle",
            2: "Car",
            3: "Motorbike",
            4: "Bus",
            5: "Truck"
        }

    def _get_color(self, track_id):
        """
        Lấy mã màu ngẫu nhiên nhưng cố định cho từng định danh (ID).

        Đảm bảo một đối tượng cụ thể (ví dụ: Xe ID số 5) sẽ luôn giữ đúng một 
        màu sắc xuyên suốt toàn bộ quá trình xuất hiện trong video.

        Args:
            track_id (int): Định danh (ID) duy nhất của đối tượng Tracking.

        Returns:
            tuple: Bộ 3 giá trị nguyên (B, G, R) biểu diễn màu sắc chuẩn của OpenCV.
        """
        idx = int(track_id) % 1000
        color = self.colors_palette[idx]
        # Chuyển đổi kiểu numpy int sang int python thuần túy để OpenCV không báo lỗi
        return (int(color[0]), int(color[1]), int(color[2]))

    def draw_tracks(self, image, active_tracks, frame_idx=0):
        """
        Vẽ các quỹ đạo đang hoạt động (active tracks) lên khung hình hiện tại.

        Thực hiện vẽ hai lớp thông tin: hộp bao thô từ YOLO (mỏng, màu vàng) nếu có 
        dữ liệu, và hộp bao dự đoán của UKF (đậm, theo màu ID) đè lên trên cùng 
        thông tin định danh. Tự động khởi tạo VideoWriter và ghi khung hình.

        Args:
            image (numpy.ndarray): Khung hình ảnh gốc (BGR).
            active_tracks (list): Danh sách các quỹ đạo. Mỗi phần tử là một tuple chứa 
                10 thông số: (track_id, class_id, ukf_x, ukf_y, ukf_w, ukf_h, yolo_x, yolo_y, yolo_w, yolo_h).
            frame_idx (int, optional): Số thứ tự khung hình hiện tại. Mặc định là 0.

        Returns:
            numpy.ndarray: Khung hình đã được vẽ đè các thông tin trực quan.
        """
        drawn_image = image.copy()
        h_img, w_img = drawn_image.shape[:2]

        # Tách các track: có detection ở frame hiện tại (visible) và track đang mất dấu dự đoán bằng UKF (lost/coasting)
        visible_tracks = [t for t in active_tracks if t[8] > 0 and t[9] > 0]
        lost_tracks = [t for t in active_tracks if t[8] == 0 or t[9] == 0]

        cv2.putText(drawn_image, f"Frame: {frame_idx:04d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(drawn_image, f"Active Tracks: {len(visible_tracks)} (Lost UKF: {len(lost_tracks)})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 1. Vẽ các track đang mất dấu (Dự đoán bởi bộ lọc UKF)
        for track in lost_tracks:
            track_id, class_id, ukf_x, ukf_y, ukf_w, ukf_h, _, _, _, _ = track
            track_id = int(track_id)
            class_id = int(class_id)
            class_name = self.class_names.get(class_id, "Unknown")
            
            ux1, uy1 = int(round(ukf_x)), int(round(ukf_y))
            ux2, uy2 = int(round(ukf_x + ukf_w)), int(round(ukf_y + ukf_h))
            ucx, ucy = int(round(ukf_x + ukf_w / 2.0)), int(round(ukf_y + ukf_h / 2.0))
            
            # Màu cam cảnh báo cho track đang mất dấu
            lost_color = (0, 140, 255)
            cv2.rectangle(drawn_image, (ux1, uy1), (ux2, uy2), lost_color, 2)
            cv2.circle(drawn_image, (ucx, ucy), radius=4, color=(0, 0, 255), thickness=-1)
            
            label = f"[LOST UKF] {class_name} | ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_top = max(uy1, text_height + 10)
            cv2.rectangle(drawn_image, (ux1, label_top - text_height - 8), (ux1 + text_width + 6, label_top), lost_color, -1)
            cv2.putText(drawn_image, label, (ux1 + 3, label_top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Vẽ các track có detection thực tế
        for track in visible_tracks:
            # Giải nén thông tin từ tracker.py
            track_id, class_id, ukf_x, ukf_y, ukf_w, ukf_h, yolo_x, yolo_y, yolo_w, yolo_h = track

            track_id = int(track_id)
            class_id = int(class_id)
            color = self._get_color(track_id)
            class_name = self.class_names.get(class_id, "Unknown")

            # ==========================================
            # VẼ HỘP BAO CHÍNH THỨC CỦA YOLO (Có Class & ID)
            # ==========================================
            yx1, yy1 = int(round(yolo_x)), int(round(yolo_y))
            yx2, yy2 = int(round(yolo_x + yolo_w)), int(round(yolo_y + yolo_h))

            # 1. Vẽ hộp bao YOLO theo màu của Track ID (độ dày viền 2px)
            cv2.rectangle(drawn_image, (yx1, yy1), (yx2, yy2), color, 2)

            # 2. Vẽ Banner nhãn hiển thị: [Class] | ID: [ID]
            label = f"{class_name} | ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Chống tràn banner lên mép trên ảnh
            label_top = max(yy1, text_height + 10)
            cv2.rectangle(drawn_image, (yx1, label_top - text_height - 10), (yx1 + text_width, label_top), color, -1)
            cv2.putText(drawn_image, label, (yx1, label_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w_img, h_img))
        
        self.writer.write(drawn_image)
        return drawn_image
    
    def draw_raw_detections(self, image, detections, frame_idx=0):
        """
        Vẽ các hộp nhận diện (bounding box) thô trực tiếp từ YOLO.

        Chức năng này bỏ qua quá trình của bộ lọc Kalman, thường được dùng để 
        kiểm thử (debug) hoặc đánh giá độ nhạy của mạng nhận diện một cách độc lập.

        Args:
            image (numpy.ndarray): Khung hình ảnh gốc (BGR).
            detections (list hoặc numpy.ndarray): Danh sách các hộp nhận diện. Mỗi phần tử 
                chứa thông số [x, y, w, h, conf, class_id].
            frame_idx (int, optional): Số thứ tự khung hình hiện tại. Mặc định là 0.

        Returns:
            numpy.ndarray: Khung hình đã được vẽ các hộp nhận diện thô (dạng hộp xám).
        """
        draw_image = image.copy()
        for det in detections:
            x, y, w, h, conf, class_id = det[:6]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            class_name = self.class_names.get(int(class_id), "Obj")

            # Box màu xám mờ để phân biệt với box màu sắc sặc sỡ của hệ thống Tracking
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), (128, 128, 128), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(draw_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
        return draw_image
    
    def release(self):
        """
        Đóng gói và xuất tệp Video hoàn chỉnh.

        Hàm này gọi phương thức release của OpenCV VideoWriter. Rất quan trọng 
        để giải phóng bộ đệm (buffer) ở cuối chu trình, nếu quên gọi hàm này, 
        tệp video lưu ra có thể bị hỏng (corrupted).
        """
        if self.writer is not None:
            self.writer.release()
            print(f"[INFO] Đã lưu video kết quả Tracking tại: {self.output_path}")