import time
import os
import numpy as np
import cv2

from utils.kitti_parser import KittiParser
from models.yolo_detector import YOLODetector
from models.reid_extractor import ReIDExtractor
from core.tracker import Tracker
from core.cmc import SparseOpticalFlowCMC

def main():
    print("=== PIPELINE EVALUATION: DEEPSORT + UKF + CMC (ALL SEQUENCES) ===")
    print("[!] Chế độ: CHỈ TRACKING NGƯỜI (0) VÀ Ô TÔ (2)")
    
    # 1. Cấu hình các tham số đường dẫn
    BASE_DATA_DIR = "datasets/KITTI_MOT"
    OUTPUT_BASE_DIR = "outputs/results"
    
    SCALE_FACTOR = 1.0 
    NMS_THRESHOLD = 0.4 
    CONF_THRESHOLD = 0.55
    FPS = 10 

    TARGET_CLASSES = [0, 2]

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Tự động quét và lấy danh sách TẤT CẢ các thư mục sequence (VD: KITTI-0000, KITTI-0001...)
    if not os.path.exists(BASE_DATA_DIR):
        print(f"❌ Lỗi: Không tìm thấy thư mục dữ liệu {BASE_DATA_DIR}")
        return
        
    sequences = sorted([d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d)) and d.startswith("KITTI-")])
    
    if not sequences:
        print(f"❌ Lỗi: Không tìm thấy sequence nào trong {BASE_DATA_DIR}")
        return

    print(f"[i] Đã tìm thấy {len(sequences)} sequences: {sequences}")

    # 2. Khởi tạo các Mạng AI (Tối ưu: Chỉ khởi tạo 1 lần ngoài vòng lặp)
    print("\n[+] Đang tải các mô hình AI lên bộ nhớ...")
    detector = YOLODetector(model_path="weights/yolo11n_int8_openvino_model", conf_thresh=CONF_THRESHOLD)
    reid_vehicle = ReIDExtractor(model_path="weights/public/vehicle-reid-0001/osnet_int8/osnet_vehicle_int8.xml", device="CPU")
    reid_person = ReIDExtractor(model_path="weights/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml", device="CPU")
    print("[+] Tải mô hình thành công!\n")

    total_pipeline_time = 0.0
    total_pipeline_frames = 0

    # 3. VÒNG LẶP XỬ LÝ TỪNG SEQUENCE
    for seq_name in sequences:
        SEQ_DIR = os.path.join(BASE_DATA_DIR, seq_name)
        OUTPUT_RESULT_FILE = os.path.join(OUTPUT_BASE_DIR, f"{seq_name}_results.txt")
        
        print("="*60)
        print(f"[+] ĐANG XỬ LÝ SEQUENCE: {seq_name}")
        print("="*60)
        
        # BẮT BUỘC: Khởi tạo lại bộ nhớ của Tracker, CMC và Parser cho MỖI video mới
        parser = KittiParser(seq_dir=SEQ_DIR)
        cmc_engine = SparseOpticalFlowCMC()
        tracker = Tracker(max_age=30, n_init=3, cosine_threshold=0.35, iou_threshold=0.7, fps=FPS)
        
        results = [] 
        seq_time = 0.0
        seq_frames = 0

        # Vòng lặp từng frame trong 1 sequence
        for frame_idx, image, _ in parser.get_frame():
            frame_start_time = time.time()

            if SCALE_FACTOR != 1.0:
                image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
                
            img_h, img_w = image.shape[:2]
                
            # Bước 1: Tính CMC
            H_camera = cmc_engine.apply(image)

            # Bước 2: Phát hiện vật thể (YOLO)
            bboxes, confs, class_ids = detector.detect(image)
            
            # NMS Filter & Class Filter
            valid_bboxes, valid_confs, valid_class_ids = [], [], []
            if len(bboxes) > 0:
                indices = cv2.dnn.NMSBoxes(bboxes, confs, score_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD)
                if len(indices) > 0:
                    indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                    for i in indices:
                        cls_id = class_ids[i]
                        if cls_id in TARGET_CLASSES:
                            valid_bboxes.append(bboxes[i])
                            valid_confs.append(confs[i])
                            valid_class_ids.append(cls_id)

            # Bước 3: Trích xuất đặc trưng ReID
            features = []
            if len(valid_bboxes) > 0:
                for i, bbox in enumerate(valid_bboxes):
                    cls_id = valid_class_ids[i]
                    if cls_id == 0: 
                        feat = reid_person.extract(image, np.array([bbox]))[0]
                        if feat.shape[0] < 512:
                            feat = np.pad(feat, (0, 512 - feat.shape[0]), 'constant')
                    elif cls_id == 2: 
                        feat = reid_vehicle.extract(image, np.array([bbox]))[0]
                        
                    features.append(feat)
                features = np.array(features, dtype=np.float32)
            else:
                features = np.zeros((0, 512), dtype=np.float32)

            # Bước 4: Cập nhật Tracker
            tracker.update(valid_bboxes, valid_confs, valid_class_ids, features, frame_shape=(img_h, img_w), H_camera=H_camera, frame_idx=frame_idx)

            # Bước 5: Thu thập kết quả
            draw_data = tracker.get_results()
            for data in draw_data:
                track_id, class_id, x, y, w, h, *_ = data
                res_line = f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,{class_id},-1,-1\n"
                results.append(res_line)

            process_time = time.time() - frame_start_time
            seq_time += process_time
            seq_frames += 1

            if frame_idx % 50 == 0:
                fps_current = 1.0 / process_time if process_time > 0 else 0
                print(f"  [>] Frame {frame_idx:04d} | Đang theo dõi: {len(draw_data):02d} objects | FPS hiện tại: {fps_current:.1f}")

        # Lưu file kết quả cho sequence hiện tại
        with open(OUTPUT_RESULT_FILE, "w") as f:
            f.writelines(results)

        seq_avg_fps = seq_frames / seq_time if seq_time > 0 else 0
        print(f"[v] Hoàn thành {seq_name} | {seq_frames} frames | Tốc độ: {seq_avg_fps:.1f} FPS")
        print(f"[v] Lịch sử quỹ đạo lưu tại: {OUTPUT_RESULT_FILE}\n")

        total_pipeline_time += seq_time
        total_pipeline_frames += seq_frames

    # TỔNG KẾT TOÀN BỘ DATASET
    total_avg_fps = total_pipeline_frames / total_pipeline_time if total_pipeline_time > 0 else 0
    print("============================================")
    print("=== HOÀN THÀNH INFERENCE TRÊN TOÀN BỘ KITTI ===")
    print(f"Tổng số Video đã chạy    : {len(sequences)}")
    print(f"Tổng số Frame đã xử lý   : {total_pipeline_frames}")
    print(f"Tổng thời gian chạy      : {total_pipeline_time:.2f} giây")
    print(f"Tốc độ trung bình (FPS)  : {total_avg_fps:.1f} FPS")
    print("============================================")

if __name__ == "__main__":
    main()