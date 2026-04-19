import cv2
import numpy as np
import time
from utils.kitti_parser import KittiParser
from utils.visualizer import Visualizer
from models.yolo_detector import YOLODetector
from models.reid_extractor import ReIDExtractor
from core.tracker import Tracker
from core.cmc import SparseOpticalFlowCMC

def main():
    print("=== PIPELINE: DEEPSORT + UKF + CMC (OOP REFACTORED) ===")
    
    CONF_THRESHOLD = 0.6
    FPS = 10

    # Khởi tạo mô-đun với đường dẫn và cấu hình từ file cũ
    parser = KittiParser(seq_dir="datasets/KITTI_MOT/KITTI-0001")
    visualizer = Visualizer(output_path="outputs/videos/test_ukf_cmc.mp4", fps=FPS)
    
    detector = YOLODetector(model_path="weights/yolo11n_int8_openvino_model", conf_thresh=CONF_THRESHOLD)
    reid_person = ReIDExtractor(model_path="weights/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml", device="CPU")
    reid_vehicle = ReIDExtractor(model_path="weights/public/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.xml", device="CPU")
    
    cmc_engine = SparseOpticalFlowCMC()
    tracker = Tracker(max_age=30, n_init=3, cosine_threshold=0.35, high_thresh=CONF_THRESHOLD)

    # Đo lường tổng thể
    total_time = 0.0
    frame_count = 0

    for frame_idx, image, _ in parser.get_frame():
        start_time = time.perf_counter()

        # 1. Bù trừ chuyển động nền (Camera Motion Compensation)
        H_camera = cmc_engine.apply(image)

        # 2. Xử lý nhận diện đối tượng
        valid_bboxes, valid_confs, valid_class_ids = detector.detect(image)
        
        # Khởi tạo mảng tính năng cho TẤT CẢ bounding boxes (bao gồm cả giá trị rỗng)
        features = np.zeros((len(valid_bboxes), 512), dtype=np.float32)
        
        # 3. Phân chia chiến lược Trích xuất (BYTE-TRACK Logic) 
        # Chỉ chạy mô hình sâu (Deep Model) trên các detections đạt độ tin cậy trên CONF_THRESHOLD
        person_idx, vehicle_idx = [], []
        person_bboxes, vehicle_bboxes = [], []
        
        for i, (bbox, conf, cls_id) in enumerate(zip(valid_bboxes, valid_confs, valid_class_ids)):
            if conf >= CONF_THRESHOLD:
                if cls_id == 0:  # Định danh Person
                    person_idx.append(i)
                    person_bboxes.append(bbox)
                else:            # Định danh Vehicle
                    vehicle_idx.append(i)
                    vehicle_bboxes.append(bbox)
                    
        # Đẩy song song khối lượng lớn (Batch Execution)
        if len(person_bboxes) > 0:
            person_feats = reid_person.extract(image, person_bboxes)
            for p_id, feat in zip(person_idx, person_feats):
                # Padding zero nếu đặc trưng của person < 512 (đồng bộ với kích thước vector Tracker)
                if feat.shape[0] < 512:
                    feat = np.pad(feat, (0, 512 - feat.shape[0]), 'constant')
                features[p_id] = feat
                
        if len(vehicle_bboxes) > 0:
            vehicle_feats = reid_vehicle.extract(image, vehicle_bboxes)
            for v_id, feat in zip(vehicle_idx, vehicle_feats):
                features[v_id] = feat

        # 4. Cập nhật hệ thống Tracking 
        tracker.update(valid_bboxes, valid_confs, valid_class_ids, features, 
                       frame_shape=(image.shape[0], image.shape[1]), 
                       H_camera=H_camera, frame_idx=frame_idx)

        # 5. Phân giải và xuất kết quả
        draw_data = tracker.get_results()
        
        # Dùng phương thức vẽ của file mới hoặc cũ tùy theo class Visualizer của bạn hỗ trợ tên hàm nào
        vis_image = visualizer.draw(image, draw_data) if hasattr(visualizer, 'draw') else visualizer.draw_tracks(image, draw_data, frame_idx)
        
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        frame_count += 1
        
        # Render FPS trên hình
        fps_display = frame_count / total_time
        cv2.putText(vis_image, f"FPS: {fps_display:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        cv2.imshow("Multi-Object Tracking KITTI", vis_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Dọn dẹp tài nguyên
    if hasattr(visualizer, 'release'):
        visualizer.release()
    cv2.destroyAllWindows()
    print("=== HOÀN THÀNH ===")

if __name__ == "__main__":
    main()