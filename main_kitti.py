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

    parser = KittiParser(seq_dir="datasets/KITTI_MOT/KITTI-0001")
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

        # Bước 3: Trích xuất đặc trưng ReID
        t_reid_start = time.time()
        features = np.zeros((len(valid_bboxes), 512), dtype=np.float32)
        
        # (Bạn có thể phân loại gọi extractor vehicle/person tại đây)
        if len(valid_bboxes) > 0:
            features = reid_vehicle.extract(image, np.array(valid_bboxes))

        # Bước 4: Đẩy hết dữ liệu cho Tracker xử lý (Tracker tự làm 3 vòng match)
        tracker.update(valid_bboxes, valid_confs, valid_class_ids, features, frame_shape=(img_h, img_w), H_camera=H_camera)
        
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