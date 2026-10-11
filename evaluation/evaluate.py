"""
Kịch bản đánh giá toàn cục (Evaluation Pipeline) trên tập dữ liệu KITTI.

Mô đun này thực thi quy trình theo dõi đa mục tiêu (Multi-Object Tracking) bằng cách 
kết hợp YOLO (Detection), OSNet/Intel ReID (Feature Extraction), Sparse Optical Flow 
(Camera Motion Compensation), và Unscented Kalman Filter (Tracking). 

Hệ thống sẽ tự động quét qua toàn bộ các chuỗi video (sequences) trong thư mục KITTI, 
chạy thuật toán và xuất kết quả quỹ đạo ra các tệp văn bản theo chuẩn MOT Challenge.
"""

import time
import os
import sys
import numpy as np
import cv2

# Đảm bảo import được các module từ thư mục gốc của dự án
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataloader.dataloader import KittiParser
from src.detector.detector import YOLODetector
from src.extractor.extractor import ReIDExtractor
from src.tracker.tracker import Tracker
from src.cmc.cmc import SparseOpticalFlowCMC

def main():
    """
    Hàm thực thi chính của chương trình đánh giá (Inference).

    Quy trình hoạt động:
    1. Cấu hình tham số môi trường và đường dẫn thư mục.
    2. Tự động quét và lập danh sách các chuỗi video KITTI cần xử lý.
    3. Khởi tạo một lần duy nhất các mô hình AI (YOLO, ReID) lên bộ nhớ phần cứng.
    4. Duyệt qua từng chuỗi video:
        - Đặt lại trạng thái bộ nhớ của Tracker, CMC và KittiParser để tránh chồng chéo.
        - Xử lý từng khung hình: Áp dụng CMC -> Nhận diện (YOLO) -> Trích xuất đặc trưng -> Cập nhật UKF.
        - Định dạng kết quả quỹ đạo thành chuỗi ký tự theo chuẩn MOT16.
    5. Ghi tập hợp kết quả của từng chuỗi ra tệp văn bản tương ứng.
    6. Tính toán và in ra báo cáo tổng kết hiệu năng tổng thể (FPS, thời gian).
    """
    print("[INFO] === PIPELINE EVALUATION: DEEPSORT + UKF + CMC (ALL SEQUENCES) ===")
    print("[INFO] Chế độ: CHỈ TRACKING NGƯỜI (0) VÀ Ô TÔ (2)")
    
    # 1. Cấu hình các tham số đường dẫn và thuật toán
    BASE_DATA_DIR = "datasets/KITTI_MOT"
    OUTPUT_BASE_DIR = "outputs/results"
    
    SCALE_FACTOR = 1.0 
    NMS_THRESHOLD = 0.4 
    CONF_THRESHOLD = 0.55
    FPS = 10 

    TARGET_CLASSES = [0, 2]

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Tự động quét và lấy danh sách tất cả các thư mục sequence (VD: KITTI-0000, KITTI-0001...)
    if not os.path.exists(BASE_DATA_DIR):
        print(f"[ERROR] Không tìm thấy thư mục dữ liệu: {BASE_DATA_DIR}")
        return
        
    sequences = sorted([d for d in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, d)) and d.startswith("KITTI-")])
    
    if not sequences:
        print(f"[ERROR] Không tìm thấy sequence nào trong: {BASE_DATA_DIR}")
        return

    print(f"[INFO] Đã tìm thấy {len(sequences)} sequences: {sequences}")

    # 2. Khởi tạo các mạng AI (Tối ưu: Chỉ khởi tạo 1 lần ngoài vòng lặp để tránh tràn RAM)
    print("\n[INFO] Đang tải các mô hình AI lên bộ nhớ...")
    detector = YOLODetector(model_path="weights/yolo26n.pt", conf_thresh=CONF_THRESHOLD)
    reid = ReIDExtractor(model_path="weights/unified/unified_reid_int8.xml", device="CPU")
    print("[SUCCESS] Tải mô hình thành công!\n")

    total_pipeline_time = 0.0
    total_pipeline_frames = 0

    # 3. VÒNG LẶP XỬ LÝ TỪNG SEQUENCE
    for seq_name in sequences:
        SEQ_DIR = os.path.join(BASE_DATA_DIR, seq_name)
        OUTPUT_RESULT_FILE = os.path.join(OUTPUT_BASE_DIR, f"{seq_name}_results.txt")
        
        print("="*60)
        print(f"[INFO] ĐANG XỬ LÝ SEQUENCE: {seq_name}")
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
                
            # Bước 1: Tính toán bù trừ chuyển động máy ảnh (CMC)
            H_camera = cmc_engine.apply(image)

            # Bước 2: Phát hiện vật thể (YOLO)
            bboxes, confs, class_ids = detector.detect(image)
            
            # YOLODetector đã thực hiện NMS phân biệt từng lớp và lọc sẵn Person (0) & Car (2)
            valid_bboxes = bboxes
            valid_confs = confs
            valid_class_ids = class_ids

            # Bước 3: Trích xuất đặc trưng ReID bằng Unified ReID duy nhất cho cả Người và Xe
            if len(valid_bboxes) > 0:
                features = reid.extract(image, valid_bboxes)
            else:
                features = np.zeros((0, 256), dtype=np.float32)

            # Bước 4: Đẩy dữ liệu vào Tracker (Cập nhật UKF và Liên kết dữ liệu)
            tracker.update(valid_bboxes, valid_confs, valid_class_ids, features, frame_shape=(img_h, img_w), H_camera=H_camera, frame_idx=frame_idx)

            # Bước 5: Thu thập kết quả theo chuẩn MOT16 (Chế độ Benchmark: only_detected=True)
            draw_data = tracker.get_results(only_detected=True)
            for data in draw_data:
                track_id, class_id, x, y, w, h, *_ = data
                res_line = f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,{class_id},-1,-1\n"
                results.append(res_line)

            process_time = time.time() - frame_start_time
            seq_time += process_time
            seq_frames += 1

            if frame_idx % 50 == 0:
                fps_current = 1.0 / process_time if process_time > 0 else 0
                print(f"  [INFO] Frame {frame_idx:04d} | Đang theo dõi: {len(draw_data):02d} objects | FPS hiện tại: {fps_current:.1f}")

        # Lưu file kết quả cho sequence hiện tại
        with open(OUTPUT_RESULT_FILE, "w") as f:
            f.writelines(results)

        seq_avg_fps = seq_frames / seq_time if seq_time > 0 else 0
        print(f"[SUCCESS] Hoàn thành {seq_name} | {seq_frames} frames | Tốc độ: {seq_avg_fps:.1f} FPS")
        print(f"[INFO] Lịch sử quỹ đạo lưu tại: {OUTPUT_RESULT_FILE}\n")

        total_pipeline_time += seq_time
        total_pipeline_frames += seq_frames

    # TỔNG KẾT TOÀN BỘ DATASET
    total_avg_fps = total_pipeline_frames / total_pipeline_time if total_pipeline_time > 0 else 0
    print("============================================")
    print("[SUCCESS] === HOÀN THÀNH INFERENCE TRÊN TOÀN BỘ KITTI ===")
    print(f"Tổng số Video đã chạy    : {len(sequences)}")
    print(f"Tổng số Frame đã xử lý   : {total_pipeline_frames}")
    print(f"Tổng thời gian chạy      : {total_pipeline_time:.2f} giây")
    print(f"Tốc độ trung bình (FPS)  : {total_avg_fps:.1f} FPS")
    print("============================================")

if __name__ == "__main__":
    main()