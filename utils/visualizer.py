# Vẽ Bounding Box, màu sắc, ID lên ảnh.
import cv2
import numpy as np

class Visualizer:
    """ Class hỗ trợ vẽ bounding box, ID, và các thông tin khác lên ảnh.
    """
    def __init__(self, output_path="outputs/videos/results.mp4", fps=30):
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        
        np.random.seed(42)
        self.colors_palette = np.random.randint(0, 255, size=(1000, 3), dtype=int)

    def _get_color(self, track_id):
        """ Lay màu ngẫu nhiên dựa trên track_id để dễ phân biệt các đối tượng khác nhau. """
        idx = int(track_id) % 1000
        color = self.colors_palette[idx]
        return (int(color[0]), int(color[1]), int(color[2]))
    def draw_tracks(self, image, active_tracks, frame_idx):
        """
        Vẽ danh sách các track đang hoạt động lên ảnh.
        Input:
            image: ảnh gốc (H, W, 3)
            active_tracks: List các tuple (track_id, x_min, y_min, w, h).
            frame_idx: số thứ tự in trên góc màn hình.
        Output:
            drawn_image: Ảnh đã vẽ.
        """
        # Tạo bản sao của ảnh gốc để vẽ lên đó
        drawn_image = image.copy()
        h_img, w_img = drawn_image.shape[:2]

        cv2.putText(drawn_image, f"Frame: {frame_idx:04d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(drawn_image, f"Active Tracks: {len(active_tracks)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # lap qua từng track để vẽ bounding box và ID
        for track in active_tracks:
            track_id = track[0]
            x, y, w, h = track[1:5]

            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            color = self._get_color(track_id)

            # Vẽ bounding box
            cv2.rectangle(drawn_image, (x1, y1), (x2, y2), color, 2)

            # Vẽ ID ở góc trên bên trái của bounding box
            label = f"ID: {track_id}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1   )

            # Vẽ nền cho text để dễ đọc
            cv2.rectangle(drawn_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

            # Viết chữ trắng lên nền màu
            cv2.putText(drawn_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Khởi tạo video writer nếu chưa có
        if self.writer is None:

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Tự dộng tạo thư mục output nếu chưa tồn tại
            import os
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w_img, h_img))
        
        self.writer.write(drawn_image)
        return drawn_image
    
    def draw_raw_detections(self, image, detections, frame_idx=0):
        """
        Vẽ các bounding box thô từ detections lên ảnh.
        Input: detections: mảng (N, 5) [x, y, w, h, conf]
        """
        draw_image = image.copy()
        h_img, w_img = draw_image.shape[:2]

        cv2.putText(draw_image, f"Frame: {frame_idx:04d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        for det in detections:
            x, y, w, h, conf = det[:5]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            # Vẽ bounding box màu xanh dương cho detections thô
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), (128, 128, 128), 2)
            cv2.putText(draw_image, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
        return draw_image
    
    def release(self):
        """
        Giải phóng tài nguyên video writer khi hoàn thành.
        """
        if self.writer is not None:
            self.writer.release()
            print(f"[+] Đã lưu video kết quả tại: {self.output_path}")


# Hàm test nhanh class Visualizer
if __name__ == "__main__":
    from kitti_parser import KittiParser
    import os

    test_dir = "datasets/KITTI_MOT/KITTI-0000"

    if os.path.exists(test_dir):
        print("[+] Đang test Visualizer với dữ liệu KITTI...")
        parser = KittiParser(test_dir)
        vis = Visualizer(output_path="outputs/videos/test_visualizer.mp4", fps=10)

        count = 0
        for frame_idx, img, dets in parser.get_frame():
            count += 1
            if count > 50:  # Chỉ test với 50 frame đầu tiên
                break

            print(f"Đang vẽ Frame {frame_idx}...", end='\r')

            drawn_img = vis.draw_raw_detections(img, dets, frame_idx)

            cv2.imshow("Test Visualizer", drawn_img)
            if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để dừng sớm
                break

        print("\n[+] Đã hoàn thành test Visualizer.")
        vis.release()
        cv2.destroyAllWindows()
    else:
        print(f"[!] Không tìm thấy thư mục test: {test_dir}. Vui lòng kiểm tra lại đường dẫn.")
    