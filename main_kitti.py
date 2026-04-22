"""
Kịch bản kiểm thử luồng thực thi (Test Pipeline) trên một chuỗi video đơn lẻ.

Mô đun này kết hợp toàn bộ các thành phần của hệ thống: YOLO (Nhận diện), 
ReID (Trích xuất đặc trưng ngoại hình), Sparse Optical Flow (Bù trừ chuyển động), 
và Unscented Kalman Filter (Theo dõi quỹ đạo). 
Đồng thời, tích hợp module Visualizer để hiển thị trực tiếp và lưu video kết quả,
kèm theo logic in ra các thông số nội bộ (Debug) của UKF ở các khung hình chỉ định.
"""

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
    """
    Hàm thực thi chính của chương trình kiểm thử.

    Quy trình thực hiện:
    1. Khởi tạo cấu hình và nạp các mô hình AI (YOLO, OSNet, Intel ReID).
    2. Khởi tạo lõi toán học (CMC, Tracker) và công cụ hiển thị (Visualizer).
    3. Xử lý tuần tự từng khung hình qua 5 bước: CMC -> YOLO -> ReID -> UKF -> Vẽ.
    4. Ghi nhận và in ra thời gian trễ (latency) của từng module ở mỗi khung hình.
    5. Hiển thị luồng video thời gian thực và đóng gói tệp video khi kết thúc.
    """
    print("[INFO] === PIPELINE: DEEPSORT + UKF + CMC (OOP REFACTORED) ===")
    
    SCALE_FACTOR = 1.0 
    NMS_THRESHOLD = 0.4 
    CONF_THRESHOLD = 0.55
    FPS = 10 

    # Khởi tạo đọc dữ liệu và công cụ trực quan hóa
    parser = KittiParser(seq_dir="datasets/KITTI_MOT/KITTI-0001")
    visualizer = Visualizer(output_path="outputs/videos/test_ukf_cmc.mp4", fps=FPS)
    
    # Khởi tạo các mô hình Deep Learning
    detector = YOLODetector(model_path="weights/yolo11n_int8_openvino_model", conf_thresh=CONF_THRESHOLD)
    reid_vehicle = ReIDExtractor(model_path="weights/public/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.xml", device="CPU")
    reid_person = ReIDExtractor(model_path="weights/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml", device="CPU")
    
    # Khởi tạo công cụ đo chuyển động máy ảnh
    cmc_engine = SparseOpticalFlowCMC()
    
    # Khởi tạo lõi theo dõi quỹ đạo
    tracker = Tracker(max_age=30, n_init=3, cosine_threshold=0.35, iou_threshold=0.7, fps=FPS)
    
    VEHICLE_CLASSES = [2, 3, 5, 7] 
    
    for frame_idx, image, _ in parser.get_frame():
        frame_start_time = time.time()

        if SCALE_FACTOR != 1.0:
            image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            
        img_h, img_w = image.shape[:2]
            
        # Bước 1: Tính toán bù trừ chuyển động (CMC)
        t_cmc_start = time.time()
        H_camera = cmc_engine.apply(image)
        cmc_time = (time.time() - t_cmc_start) * 1000

        # Bước 2: Phát hiện vật thể bằng YOLO
        t_yolo_start = time.time()
        bboxes, confs, class_ids = detector.detect(image)
        
        # Áp dụng bộ lọc Non-Maximum Suppression (NMS)
        valid_bboxes, valid_confs, valid_class_ids = [], [], []
        if len(bboxes) > 0:
            indices = cv2.dnn.NMSBoxes(bboxes, confs, score_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD)
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                for i in indices:
                    valid_bboxes.append(bboxes[i])
                    valid_confs.append(confs[i])
                    valid_class_ids.append(class_ids[i])
        yolo_time = (time.time() - t_yolo_start) * 1000

        # Bước 3: Trích xuất đặc trưng ReID phân cụm theo lớp đối tượng
        t_reid_start = time.time()
        features = []
        
        if len(valid_bboxes) > 0:
            for i, bbox in enumerate(valid_bboxes):
                cls_id = valid_class_ids[i]
                
                # Áp dụng mô hình Intel ReID cho Person (Class 0)
                if cls_id == 0:
                    feat = reid_person.extract(image, np.array([bbox]))[0]
                    # Đệm mảng bằng 0 để tương thích kích thước vector 512 của Tracker
                    if feat.shape[0] < 512:
                        feat = np.pad(feat, (0, 512 - feat.shape[0]), 'constant')
                # Áp dụng mô hình OSNet cho Phương tiện (Vehicle)
                else:
                    feat = reid_vehicle.extract(image, np.array([bbox]))[0]
                    
                features.append(feat)
                
            features = np.array(features, dtype=np.float32)
        else:
            features = np.zeros((0, 512), dtype=np.float32)
        time_reid_total = (time.time() - t_reid_start) * 1000

        # Bước 4: Đẩy dữ liệu vào Tracker (Lọc UKF + Ghép cặp Hungarian)
        t_tracker_start = time.time()
        tracker.update(valid_bboxes, valid_confs, valid_class_ids, features, frame_shape=(img_h, img_w), H_camera=H_camera, frame_idx=frame_idx)
        tracker_time = (time.time() - t_tracker_start) * 1000

        # =====================================================================
        # LOGIC IN RA MÀN HÌNH GỠ LỖI (DEBUG TRẠNG THÁI UKF)
        # Chỉ theo dõi và in log chi tiết từ frame 35 đến 65
        # =====================================================================
        if 35 <= frame_idx <= 65:
            print(f"[DEBUG] --- Đang phân tích nội bộ Frame {frame_idx} ---")
            found_ids = []
            
            for trk in tracker.tracks:
                # Chỉ soi chiếu chi tiết cho đối tượng có ID là 2 và 9
                if trk.track_id in [2, 9]:
                    found_ids.append(trk.track_id)
                    if trk.state == 1: state_str = "TENTATIVE (Chờ duyệt)"
                    elif trk.state == 2: state_str = "CONFIRMED (Đang bám)"
                    else: state_str = "DELETED (Đã xóa)"
                    
                    ukf_x, ukf_y, ukf_w, ukf_h = trk.to_tlwh()
                    vx, vy, omega, vh = trk.ukf.mean[4], trk.ukf.mean[5], trk.ukf.mean[6], trk.ukf.mean[7]
                    
                    print(f"  [>] ID: {trk.track_id} | State: {state_str}")
                    print(f"      - Mất dấu: {trk.time_since_update} frames | Hộp bao: X:{ukf_x:.1f}, Y:{ukf_y:.1f}, W:{ukf_w:.1f}, H:{ukf_h:.1f}")
                    print(f"      - Động học: vx={vx:.2f}, vy={vy:.2f} | Cua(omega)={omega:.4f} | Giãn nở(vh)={vh:.2f}")
                    
            for target_id in [2, 9]:
                if target_id not in found_ids:
                    print(f"  [!] ID: {target_id} | Đã BỊ XÓA HẲN khỏi bộ nhớ hệ thống.")
        # =====================================================================
        
        # Bước 5: Thu thập kết quả và Vẽ khung hình
        t_draw_start = time.time()
        draw_data = tracker.get_results()
        drawn_img = visualizer.draw_tracks(image, draw_data, frame_idx)
        draw_time = (time.time() - t_draw_start) * 1000
        
        # Đo lường tổng thời gian và tính toán FPS
        total_time = time.time() - frame_start_time
        fps = 1.0 / total_time
        
        print(f"[INFO] Frame: {frame_idx:04d} | Object: {len(draw_data):02d} | CMC: {cmc_time:4.1f}ms | YOLO: {yolo_time:4.1f}ms | ReID: {time_reid_total:4.1f}ms | UKF: {tracker_time:4.1f}ms | Draw: {draw_time:4.1f}ms | FPS: {fps:4.1f}")
        
        # Hiển thị trực tiếp (Real-time GUI)
        cv2.imshow("Test Pipeline UKF + CMC", drawn_img)
        # Nhấn phím 'ESC' (Mã 27) để thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Giải phóng tài nguyên bộ nhớ
    visualizer.release()
    cv2.destroyAllWindows()
    print("[SUCCESS] === HOÀN THÀNH QUÁ TRÌNH KIỂM THỬ ===")

if __name__ == "__main__":
    main()