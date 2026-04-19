import time
import cv2
import numpy as np
from ultralytics import YOLO

from utils.kitti_parser import KittiParser
from utils.visualizer import Visualizer

def main():
    print("=== PIPELINE: TEST YOLO PYTORCH GỐC (.pt) VỚI FPS ===")
    
    CONF_THRESHOLD = 0.55
    FPS_OUT = 10 

    fps_history = []

    parser = KittiParser(seq_dir="datasets/KITTI_MOT/KITTI-0000")
    visualizer = Visualizer(output_path="outputs/videos/test_yolo11s_pt.mp4", fps=FPS_OUT)
    
    print("Đang load mô hình PyTorch gốc (weights/yolo11s.pt)...")
    model = YOLO("weights/yolo11s.pt")
    
    coco_to_id = {
        0: 0,  
        1: 1,  
        2: 2,  
        3: 1,  
    }
    class_names = {0: "Person", 1: "Cyclist", 2: "Car"}

    for frame_idx, image, _ in parser.get_frame():
        frame_start_time = time.time()
            
        results = model(image, conf=CONF_THRESHOLD, verbose=False)[0]
        
        valid_bboxes = []
        valid_confs = []
        valid_class_ids = []

        for box in results.boxes:
            coco_cls = int(box.cls[0].item())
            
            if coco_cls not in coco_to_id:
                continue 
            
            target_id = coco_to_id[coco_cls]
            conf = float(box.conf[0].item())
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            
            valid_bboxes.append([float(x1), float(y1), float(w), float(h)])
            valid_confs.append(conf)
            valid_class_ids.append(target_id)

        # Tính toán FPS 
        process_time = time.time() - frame_start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        
        if frame_idx > 0:
            fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history) if len(fps_history) > 0 else fps
        
        print(f"--- Frame: {frame_idx:04d} | FPS: {fps:4.1f} | TB: {avg_fps:4.1f} | Phát hiện: {len(valid_bboxes)} vật thể ---")
        
        # ---------------------------------------------------------
        # VẼ FPS LÊN ẢNH GỐC (TRƯỚC KHI ĐƯA VÀO VISUALIZER)
        # ---------------------------------------------------------
        img_h, img_w = image.shape[:2]
        fps_text = f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        text_x = img_w - text_width - 20
        text_y = 40
        
        # Vẽ chữ trực tiếp lên biến `image`
        cv2.putText(image, fps_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, fps_text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        # ---------------------------------------------------------

        draw_data = []
        for i in range(len(valid_bboxes)):
            x, y, w, h = valid_bboxes[i]
            c_id = valid_class_ids[i]
            
            # Dummy ID (i + 1) để Visualizer tô màu 
            draw_data.append((i + 1, c_id, x, y, w, h))

        # Đưa ảnh (đã có chữ FPS) vào Visualizer để nó vẽ thêm BBox và tự động write video
        drawn_img = visualizer.draw_tracks(image, draw_data, frame_idx)
        
        cv2.imshow("Test YOLO PyTorch", drawn_img)
        if cv2.waitKey(1) & 0xFF == 27:  
            break

    visualizer.release()
    cv2.destroyAllWindows()
    print("=== HOÀN THÀNH LƯU VIDEO TẠI: outputs/videos/test_yolo11s_pt.mp4 ===")

if __name__ == "__main__":
    main()