import os  # SỬA: Đưa import os lên đỉnh file theo chuẩn PEP-8
import cv2
import numpy as np

class Visualizer:
    """ Class hỗ trợ vẽ bounding box, ID, tên class và lưu video kết quả."""
    def __init__(self, output_path="outputs/videos/results.mp4", fps=30):
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        
        # Tạo bảng màu cố định ngẫu nhiên cho 1000 ID đầu tiên
        np.random.seed(42)
        self.colors_palette = np.random.randint(0, 255, size=(1000, 3), dtype=int)
        
        # Từ điển ánh xạ từ Class ID sang String
        self.class_names = {0: "Person", 1: "Cyclist", 2: "Car"}

    def _get_color(self, track_id):
        """ 
        Lấy màu ngẫu nhiên nhưng cố định cho từng ID. 
        Xe ID số 5 sẽ luôn có màu xanh, dù xuất hiện ở bất kỳ frame nào.
        """
        idx = int(track_id) % 1000
        color = self.colors_palette[idx]
        # Chuyển đổi kiểu numpy int sang int python thuần túy để OpenCV không báo lỗi
        return (int(color[0]), int(color[1]), int(color[2]))

    def draw_tracks(self, image, active_tracks, frame_idx=0):
        """
        Vẽ danh sách các track đang hoạt động (Confirmed) lên ảnh.
        Input:
            image: ảnh gốc BGR shape (H, W, 3)
            active_tracks: List các tuple SÁU phần tử (track_id, class_id, x_min, y_min, w, h).
            frame_idx: số thứ tự in trên góc màn hình để dễ debug.
        Output:
            drawn_image: Ảnh đã vẽ đè.
        """
        # Tạo bản sao sâu (copy) để không làm hỏng dữ liệu ảnh gốc trong RAM
        drawn_image = image.copy()
        h_img, w_img = drawn_image.shape[:2]

        # Vẽ bảng thông tin tổng quan góc trái trên cùng
        cv2.putText(drawn_image, f"Frame: {frame_idx:04d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(drawn_image, f"Active Tracks: {len(active_tracks)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Lặp qua từng phương tiện đang chạy
        for track in active_tracks:
            # SỬA LỖI ĐỒNG BỘ: Ép giải nén chuẩn Tuple 6 phần tử thay vì slicing [1:5]
            track_id, class_id, x, y, w, h = track

            # Chuyển đổi định dạng w, h thành tọa độ (x2, y2) cho OpenCV
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            color = self._get_color(track_id)
            class_name = self.class_names.get(int(class_id), "Unknown")

            # 1. Vẽ bounding box
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)

            # 2. Tạo nội dung Label (SỬA: Hiển thị cả Class Name và ID)
            label = f"{class_name} | ID: {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # 3. Vẽ khối nền màu đặc cho Label để chữ không bị chìm vào background
            cv2.rectangle(drawn_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

            # 4. Viết chữ màu trắng đè lên khối nền màu
            cv2.putText(drawn_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # --- Khối logic lưu Video ---
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Tự động tạo thư mục output nếu người dùng quên chưa tạo
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w_img, h_img))
        
        self.writer.write(drawn_image)
        return drawn_image
    
    def draw_raw_detections(self, image, detections, frame_idx=0):
        """
        Vẽ các bounding box thô từ YOLO (trước khi qua Kalman Filter).
        Dùng để Test hệ thống nhận diện độc lập xem có nhạy không.
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
        Đóng gói và xuất file Video. (Rất quan trọng, nếu quên gọi hàm này video sẽ bị lỗi hỏng file).
        """
        if self.writer is not None:
            self.writer.release()
            print(f"[+] Đã lưu video kết quả Tracking tại: {self.output_path}")