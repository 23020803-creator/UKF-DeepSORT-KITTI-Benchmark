import time
import cv2
import numpy as np
import os

from utils.kitti_parser import KittiParser
from utils.visualizer import Visualizer

def load_ground_truth(gt_path):
    """
    Đọc file nhãn Ground Truth.
    Sửa lỗi: Đọc bằng định dạng MOT CSV (phân tách bằng dấu phẩy).
    Cột chuẩn MOT: [frame, track_id, x, y, w, h, conf, class_id, ...]
    """
    gt_data = {}
    
    if not os.path.exists(gt_path):
        print(f"❌ Lỗi: Không tìm thấy file GT tại {gt_path}")
        return gt_data
        
    # Đọc bằng numpy tương tự như trong class KittiParser của bạn
    raw_detections = np.loadtxt(gt_path, delimiter=',')
    
    # Từ điển ánh xạ Class ID sang tên (giống trong Visualizer)
    class_map = {0: "Person", 1: "Cyclist", 2: "Car"}
    
    for row in raw_detections:
        frame_id = int(row[0])
        track_id = int(row[1])
        x = float(row[2])
        y = float(row[3])
        w = float(row[4])
        h = float(row[5])
        
        # KittiParser của bạn lưu class_id ở cột số 8 (index 7)
        class_id = int(row[7]) if len(row) > 7 else 2
        class_name = class_map.get(class_id, "Unknown")
        
        if frame_id not in gt_data:
            gt_data[frame_id] = []
            
        gt_data[frame_id].append({
            'track_id': track_id,
            'class_id': class_id,
            'class_name': class_name,
            'bbox': [x, y, w, h]
        })
            
    print(f"✅ Đã tải nhãn Ground Truth từ {gt_path} ({len(gt_data)} frames có chứa vật thể)")
    return gt_data


def main():
    print("=== PIPELINE: TEST GROUND TRUTH ===\n")
    
    SEQUENCE = "KITTI-0000"
    DATA_DIR = f"datasets/KITTI_MOT/{SEQUENCE}"
    GT_FILE = os.path.join(DATA_DIR, "gt/gt.txt") 
    OUTPUT_VIDEO = f"outputs/videos/test_gt_{SEQUENCE}.mp4"
    FPS = 10 

    # Khởi tạo Parser và Visualizer
    parser = KittiParser(seq_dir=DATA_DIR)
    visualizer = Visualizer(output_path=OUTPUT_VIDEO, fps=FPS)
    
    # Đọc dữ liệu
    print(f"Đang đọc dữ liệu GT từ: {GT_FILE}")
    gt_data = load_ground_truth(GT_FILE)

    if not gt_data:
        print("Không có dữ liệu GT. Dừng chương trình.")
        return

    print("\nBắt đầu xử lý Video...")

    # Duyệt qua từng khung hình
    for frame_idx, image, _ in parser.get_frame():
        objects_in_frame = gt_data.get(frame_idx, [])
        
        print(f"\n--- Frame: {frame_idx:04d} | Có {len(objects_in_frame)} vật thể (GT) ---")
        
        draw_data = []
        for obj in objects_in_frame:
            t_id = obj['track_id']
            c_id = obj['class_id']
            c_name = obj['class_name']
            x, y, w, h = obj['bbox']
            
            print(f"  -> Track ID {t_id:3d}: {c_name:10s} (Class: {c_id}) | BBox: [x: {x:6.1f}, y: {y:6.1f}, w: {w:5.1f}, h: {h:5.1f}]")
            
            draw_data.append((t_id, c_id, x, y, w, h))

        # Vẽ và lưu vào video
        drawn_img = visualizer.draw_tracks(image, draw_data, frame_idx)
        cv2.imshow(f"Test Ground Truth - {SEQUENCE}", drawn_img)
        
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    visualizer.release()
    cv2.destroyAllWindows()
    print(f"\n=== HOÀN THÀNH LƯU VIDEO TẠI: {OUTPUT_VIDEO} ===")

if __name__ == "__main__":
    main()