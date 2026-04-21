import os
import sys
import shutil
import numpy as np

# --- VÁ LỖI PHIÊN BẢN NUMPY ---
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object
# ------------------------------

try:
    import trackeval
except ImportError:
    print("❌ Lỗi: Chưa cài đặt thư viện TrackEval.")
    sys.exit(1)

def evaluate_single_class(target_class_name, target_mot_id):
    print("\n" + "="*60)
    print(f" 📊 BẢNG ĐIỂM CHỈ SỐ CHO ĐỐI TƯỢNG: {target_class_name.upper()}")
    print("="*60)

    # 1. Đường dẫn gốc
    BASE_GT_DIR = "datasets/KITTI_MOT"
    BASE_TRK_DIR = "outputs/results"
    
    if not os.path.exists(BASE_TRK_DIR):
        print(f"❌ Không tìm thấy thư mục kết quả: {BASE_TRK_DIR}")
        return

    # Lấy danh sách tất cả các sequence đã được chạy Tracking
    sequences = sorted([f.replace("_results.txt", "") for f in os.listdir(BASE_TRK_DIR) if f.endswith("_results.txt")])
    
    if not sequences:
        print("⚠️ Không có file kết quả tracking nào để chấm điểm.")
        return

    # Tạo Workspace ĐỘC LẬP cho từng class để không bị đụng độ
    workspace = f"TrackEval_Workspace_{target_class_name}"
    if os.path.exists(workspace):
        shutil.rmtree(workspace) # Xóa cũ đi làm lại cho sạch

    valid_sequences = [] # Lưu trữ các sequence có chứa đối tượng đang xét

    for seq in sequences:
        result_file = os.path.join(BASE_TRK_DIR, f"{seq}_results.txt")
        gt_dir = os.path.join(BASE_GT_DIR, seq)
        
        if not os.path.exists(gt_dir):
            continue

        gt_dest_dir = os.path.join(workspace, "data", "gt", "mot_challenge", "KITTI_CUSTOM-train", seq)
        trk_dest_dir = os.path.join(workspace, "data", "trackers", "mot_challenge", "KITTI_CUSTOM-train", "UKF_DeepSORT", "data")
        os.makedirs(os.path.join(gt_dest_dir, "gt"), exist_ok=True)
        os.makedirs(trk_dest_dir, exist_ok=True)

        # --- A. Tạo Seqinfo ---
        seqinfo_path = os.path.join(gt_dir, "seqinfo.ini")
        if os.path.exists(seqinfo_path):
            shutil.copy(seqinfo_path, os.path.join(gt_dest_dir, "seqinfo.ini"))
        else:
            with open(os.path.join(gt_dest_dir, "seqinfo.ini"), "w") as f:
                f.write(f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=10\nseqLength=1000\nimWidth=1242\nimHeight=375\nimExt=.png\n")

        # --- B. LỌC VÀ ĐÁNH LỪA DỮ LIỆU GROUND TRUTH ---
        gt_source_file = os.path.join(gt_dir, "gt", "gt.txt")
        if not os.path.exists(gt_source_file):
            gt_source_file = os.path.join(gt_dir, "gt.txt")

        gt_lines = []
        if os.path.exists(gt_source_file):
            with open(gt_source_file, "r") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        cls_id = int(float(parts[7]))
                        if cls_id == target_mot_id:
                            parts[7] = '1' 
                            gt_lines.append(",".join(parts) + "\n")

        # --- C. LỌC VÀ ĐÁNH LỪA DỮ LIỆU TRACKER KẾT QUẢ ---
        trk_lines = []
        with open(result_file, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    yolo_cls = int(float(parts[7]))
                    mot_cls = 1 if yolo_cls == 0 else (3 if yolo_cls == 2 else yolo_cls)
                    
                    if mot_cls == target_mot_id:
                        parts[7] = '1' 
                        trk_lines.append(",".join(parts) + "\n")
                        
        # Nếu sequence có đối tượng (ở Ground Truth hoặc Tracker) thì mới ghi file và đưa vào chấm
        if len(gt_lines) > 0 or len(trk_lines) > 0:
            with open(os.path.join(gt_dest_dir, "gt", "gt.txt"), "w") as f:
                f.writelines(gt_lines)
            with open(os.path.join(trk_dest_dir, f"{seq}.txt"), "w") as f:
                f.writelines(trk_lines)
            valid_sequences.append(seq)

    if not valid_sequences:
        print(f"⚠️ Bỏ qua chấm điểm: Không tìm thấy đối tượng {target_class_name} hợp lệ trong toàn bộ dataset.")
        return

    # 5. Tạo Seqmap tổng hợp tất cả các valid sequences
    seqmap_dir = os.path.join(workspace, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, "KITTI_CUSTOM-train.txt"), "w") as f:
        f.write("name\n")
        for seq in valid_sequences:
            f.write(f"{seq}\n")

    # 6. BẮT ĐẦU CHẤM ĐIỂM BẰNG TRACKEVAL MẶC ĐỊNH
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['PRINT_RESULTS'] = True
    eval_config['PRINT_ONLY_COMBINED'] = False
    eval_config['PRINT_CONFIG'] = False
    eval_config['TIME_PROGRESS'] = False
    eval_config['OUTPUT_SUMMARY'] = True
    eval_config['OUTPUT_DETAILED'] = False # Tắt chi tiết để terminal gọn hơn
    eval_config['PLOT_CURVES'] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = os.path.join(workspace, 'data', 'gt', 'mot_challenge')
    dataset_config['TRACKERS_FOLDER'] = os.path.join(workspace, 'data', 'trackers', 'mot_challenge')
    dataset_config['OUTPUT_FOLDER'] = os.path.join(workspace, 'outputs')
    dataset_config['TRACKERS_TO_EVAL'] = ['UKF_DeepSORT']
    
    dataset_config['CLASSES_TO_EVAL'] = ['pedestrian'] 
    
    dataset_config['BENCHMARK'] = 'KITTI_CUSTOM'
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['INPUT_AS_ZIP'] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.HOTA(), 
        trackeval.metrics.CLEAR(),    
        trackeval.metrics.Identity()
    ]

    try:
        evaluator.evaluate(dataset_list, metrics_list)
    except Exception as e:
        print(f"⚠️ Lỗi khi chấm điểm {target_class_name.upper()}: {e}")

if __name__ == '__main__':
    # Chấm điểm cho Người đi bộ (MOT16 Class ID = 1)
    evaluate_single_class(target_class_name='pedestrian', target_mot_id=1)
    
    # Chấm điểm cho Xe ô tô (MOT16 Class ID = 3)
    evaluate_single_class(target_class_name='car', target_mot_id=3)
    
    print("\n=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH CHẤM ĐIỂM ===")