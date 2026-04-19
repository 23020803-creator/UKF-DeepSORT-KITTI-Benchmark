import os
import shutil
from ultralytics import YOLO

print("Đang kết nối và tải YOLO11s...")

# Khởi tạo mô hình, Ultralytics sẽ tự động tải file yolo11s.pt về thư mục hiện tại nếu chưa có
model = YOLO("yolo11s.pt")

# Đảm bảo thư mục weights tồn tại và di chuyển file vào đó cho gọn gàng
os.makedirs("weights", exist_ok=True)

if os.path.exists("yolo11s.pt"):
    shutil.move("yolo11s.pt", "weights/yolo11s.pt")
    print("✅ Đã tải xong và lưu model tại: weights/yolo11s.pt")
else:
    print("☑️ Model yolo11s.pt đã có sẵn trong máy.")