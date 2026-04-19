import time
import numpy as np
import cv2

from utils.kitti_parser import KittiParser
from utils.visualizer import Visualizer
from models.yolo_detector import YOLODetector
from models.reid_extractor import ReIDExtractor
from core.tracker import Tracker
from core.cmc import SparseOpticalFlowCMC

def main():
    print("=== PIPELINE: DEEPSORT + UKF + CMC (OOP REFACTORED) ===")
    
    SCALE_FACTOR = 1.0 
    NMS_THRESHOLD = 0.4 
    CONF_THRESHOLD = 0.55
    FPS = 10 

    parser = KittiParser(seq_dir="datasets/KITTI_MOT/KITTI-0000")
    visualizer = Visualizer(output_path="outputs/videos/test_ukf_cmc.mp4", fps=FPS)
    
    detector = YOLODetector(model_path="weights/yolo11n_int8_openvino_model", conf_thresh=CONF_THRESHOLD)
    reid_vehicle = ReIDExtractor(model_path="weights/public/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.xml", device="CPU")
    reid_person = ReIDExtractor(model_path="weights/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml", device="CPU")
    
    cmc_engine = SparseOpticalFlowCMC()
    
    # KHỞI TẠO TRACKER CHÍNH THỐNG
    tracker = Tracker(max_age=30, n_init=3, cosine_threshold=0.35, iou_threshold=0.7, fps=FPS)
    
    VEHICLE_CLASSES = [2, 3, 5, 7] 
    
    for frame_idx, image, _ in parser.get_frame():
        frame_start_time = time.time()

        if SCALE_FACTOR != 1.0:
            image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            
        img_h, img_w = image.shape[:2]
            
        # Bước 1: Tính CMC
        t_cmc_start = time.time()
        H_camera = cmc_engine.apply(image)
        cmc_time = (time.time() - t_cmc_start) * 1000

        # Bước 2: Phát hiện vật thể (YOLO)
        bboxes, confs, class_ids = detector.detect(image)
        
        # NMS Filter
        valid_bboxes, valid_confs, valid_class_ids = [], [], []
        if len(bboxes) > 0:
            indices = cv2.dnn.NMSBoxes(bboxes, confs, score_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD)
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                for i in indices:
                    valid_bboxes.append(bboxes[i])
                    valid_confs.append(confs[i])
                    valid_class_ids.append(class_ids[i])

        # Bước 3: Trích xuất đặc trưng ReID (Đã phân loại Class)
        t_reid_start = time.time()
        features = []
        
        if len(valid_bboxes) > 0:
            for i, bbox in enumerate(valid_bboxes):
                cls_id = valid_class_ids[i]
                
                # Trong YOLO (COCO dataset), class 0 là 'person'
                if cls_id == 0:
                    # Trích xuất đặc trưng người
                    feat = reid_person.extract(image, np.array([bbox]))[0]
                    # Đệm thêm zero để đồng nhất kích thước vector 512 của Tracker
                    if feat.shape[0] < 512:
                        feat = np.pad(feat, (0, 512 - feat.shape[0]), 'constant')
                else:
                    # Trích xuất đặc trưng xe cộ
                    feat = reid_vehicle.extract(image, np.array([bbox]))[0]
                    
                features.append(feat)
                
            features = np.array(features, dtype=np.float32)
        else:
            features = np.zeros((0, 512), dtype=np.float32)

        # Bước 4: Đẩy hết dữ liệu cho Tracker xử lý (Tracker tự làm 3 vòng match)
        tracker.update(valid_bboxes, valid_confs, valid_class_ids, features, frame_shape=(img_h, img_w), H_camera=H_camera, frame_idx=frame_idx)

        if 35 <= frame_idx <= 65:
            print(f"--- Đang phân tích Frame {frame_idx} ---")
            found_ids = []
            
            for trk in tracker.tracks:
                if trk.track_id in [2, 9]:
                    found_ids.append(trk.track_id)
                    if trk.state == 1: state_str = "TENTATIVE (Chờ duyệt)"
                    elif trk.state == 2: state_str = "CONFIRMED (Đang bám)"
                    else: state_str = "DELETED (Đã xóa)"
                    
                    ukf_x, ukf_y, ukf_w, ukf_h = trk.to_tlwh()
                    
                    # Lấy thông tin động học (vận tốc) từ vector trạng thái của UKF
                    # Vector x: [cx, cy, a, h, vx, vy, omega, vh]
                    vx = trk.ukf.mean[4]
                    vy = trk.ukf.mean[5]
                    omega = trk.ukf.mean[6]
                    vh = trk.ukf.mean[7]
                    
                    print(f"  [>] ID: {trk.track_id} | State: {state_str}")
                    print(f"      - Mất dấu (Time since update): {trk.time_since_update} frames")
                    print(f"      - Hộp dự đoán: X:{ukf_x:.1f}, Y:{ukf_y:.1f}, W:{ukf_w:.1f}, H:{ukf_h:.1f}")
                    print(f"      - Vận tốc: vx={vx:.2f}, vy={vy:.2f} | Cua(omega)={omega:.4f} | Giãn nở(vh)={vh:.2f}")
                    
            for target_id in [2, 9]:
                if target_id not in found_ids:
                    print(f"  [!] ID: {target_id} | Đã BỊ XÓA HẲN khỏi bộ nhớ.")
        # =====================================================================
        
        time_reid_total = (time.time() - t_reid_start) * 1000
        fps = 1.0 / (time.time() - frame_start_time)
        
        # Bước 5: Lấy tọa độ và Vẽ
        draw_data = tracker.get_results()
        drawn_img = visualizer.draw_tracks(image, draw_data, frame_idx)
        
        print(f"Frame: {frame_idx:04d} | Object: {len(draw_data):02d} | CMC: {cmc_time:4.1f}ms | ReID: {time_reid_total:4.1f}ms | FPS: {fps:4.1f}")
        
        cv2.imshow("Test Pipeline UKF + CMC", drawn_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    visualizer.release()
    cv2.destroyAllWindows()
    print("=== HOÀN THÀNH ===")

if __name__ == "__main__":
    main()