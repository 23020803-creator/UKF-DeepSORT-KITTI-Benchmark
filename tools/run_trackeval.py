"""
Kịch bản chấm điểm và đánh giá quỹ đạo (TrackEval Wrapper).

Mô đun này tự động hóa quy trình đánh giá kết quả tracking bằng công cụ TrackEval.
Đặc biệt, nó tích hợp cơ chế "Monkey Patch" để xử lý sự không tương thích của 
thư viện TrackEval với các phiên bản NumPy mới (vấn đề np.float).

Đồng thời, mã nguồn thực hiện quy trình "đánh lừa" dữ liệu (Data Spoofing):
Tách biệt hoàn toàn việc chấm điểm cho Người (Pedestrian) và Xe (Car) bằng cách
chủ động lọc và gán lại nhãn (class_id = 1) để vượt qua bộ kiểm duyệt cứng nhắc 
của thư viện TrackEval gốc.
"""

import os
import sys
import shutil
import numpy as np

# ==============================================================================
# VÁ LỖI TƯƠNG THÍCH PHIÊN BẢN NUMPY (MONKEY PATCHING)
# ==============================================================================
# Thư viện TrackEval gọi các thuộc tính đã bị deprecate trong NumPy >= 1.20
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object

try:
    import trackeval
except ImportError:
    print("[ERROR] Lỗi: Chưa cài đặt thư viện TrackEval.")
    print("[HINT] Chạy lệnh: pip install git+https://github.com/JonathonLuiten/TrackEval.git")
    sys.exit(1)

def evaluate_single_class(target_class_name, target_mot_id):
    """
    Thực thi việc cấu hình môi trường ảo và chạy chấm điểm cho một lớp đối tượng cụ thể.

    Hàm này tạo ra một Workspace độc lập (Temporary Directory) cho đối tượng mục tiêu.
    Nó quét các tệp nhãn gốc (Ground Truth) và tệp kết quả (Tracker Results),
    lọc ra đúng loại đối tượng cần chấm, sau đó ép nhãn của chúng về `1` (Pedestrian)
    để ép buộc TrackEval phải xử lý dữ liệu.

    Args:
        target_class_name (str): Tên chuỗi (String) của lớp đối tượng (VD: 'pedestrian', 'car').
        target_mot_id (int): Mã ID chuẩn tương ứng của đối tượng theo chuẩn MOT16 
                             (VD: 1 cho Người, 3 cho Xe ô tô).
    """
    print("\n" + "="*60)
    print(f"[INFO] BẢNG ĐIỂM CHỈ SỐ CHO ĐỐI TƯỢNG: {target_class_name.upper()}")
    print("="*60)

    # 1. Đường dẫn thư mục gốc
    BASE_GT_DIR = "datasets/KITTI_MOT"
    BASE_TRK_DIR = "outputs/results"
    
    if not os.path.exists(BASE_TRK_DIR):
        print(f"[ERROR] Không tìm thấy thư mục chứa kết quả tracking: {BASE_TRK_DIR}")
        return

    # Quét thư mục để lấy danh sách tất cả các sequence đã được chạy Tracking
    sequences = sorted([f.replace("_results.txt", "") for f in os.listdir(BASE_TRK_DIR) if f.endswith("_results.txt")])
    
    if not sequences:
        print("[WARNING] Không có tệp kết quả tracking nào để tiến hành chấm điểm.")
        return

    # Tạo Workspace ĐỘC LẬP cho từng class để dữ liệu không bị xung đột
    workspace = f"TrackEval_Workspace_{target_class_name}"
    if os.path.exists(workspace):
        shutil.rmtree(workspace) # Xóa Workspace cũ để đảm bảo dữ liệu sạch

    valid_sequences = [] # Danh sách lưu các sequence thực sự chứa đối tượng đang xét

    for seq in sequences:
        result_file = os.path.join(BASE_TRK_DIR, f"{seq}_results.txt")
        gt_dir = os.path.join(BASE_GT_DIR, seq)
        
        if not os.path.exists(gt_dir):
            continue

        gt_dest_dir = os.path.join(workspace, "data", "gt", "mot_challenge", "KITTI_CUSTOM-train", seq)
        trk_dest_dir = os.path.join(workspace, "data", "trackers", "mot_challenge", "KITTI_CUSTOM-train", "UKF_DeepSORT", "data")
        os.makedirs(os.path.join(gt_dest_dir, "gt"), exist_ok=True)
        os.makedirs(trk_dest_dir, exist_ok=True)

        # --- A. Khởi tạo tệp cấu hình Seqinfo ---
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
                            # TRICK: Ép class ID về 1 để vượt rào TrackEval
                            parts[7] = '1' 
                            gt_lines.append(",".join(parts) + "\n")

        # --- C. LỌC VÀ ĐÁNH LỪA DỮ LIỆU TRACKER KẾT QUẢ ---
        trk_lines = []
        with open(result_file, "r") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    yolo_cls = int(float(parts[7]))
                    # Ánh xạ từ YOLO Class sang MOT16 Class
                    mot_cls = 1 if yolo_cls == 0 else (3 if yolo_cls == 2 else yolo_cls)
                    
                    if mot_cls == target_mot_id:
                        # TRICK: Ép class ID về 1 để vượt rào TrackEval
                        parts[7] = '1' 
                        trk_lines.append(",".join(parts) + "\n")
                        
        # Chỉ đẩy vào hàng đợi chấm điểm nếu sequence thực sự có đối tượng xuất hiện
        if len(gt_lines) > 0 or len(trk_lines) > 0:
            with open(os.path.join(gt_dest_dir, "gt", "gt.txt"), "w") as f:
                f.writelines(gt_lines)
            with open(os.path.join(trk_dest_dir, f"{seq}.txt"), "w") as f:
                f.writelines(trk_lines)
            valid_sequences.append(seq)

    if not valid_sequences:
        print(f"[WARNING] Bỏ qua chấm điểm: Không tìm thấy đối tượng '{target_class_name}' hợp lệ trong toàn bộ dataset.")
        return

    # 5. Khởi tạo tệp Seqmap tổng hợp danh sách các valid sequences
    seqmap_dir = os.path.join(workspace, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    with open(os.path.join(seqmap_dir, "KITTI_CUSTOM-train.txt"), "w") as f:
        f.write("name\n")
        for seq in valid_sequences:
            f.write(f"{seq}\n")

    # 6. THỰC THI CHẤM ĐIỂM (TRACKEVAL EVALUATOR)
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = True
    eval_config['PRINT_RESULTS'] = True
    eval_config['PRINT_ONLY_COMBINED'] = False
    eval_config['PRINT_CONFIG'] = False
    eval_config['TIME_PROGRESS'] = False
    eval_config['OUTPUT_SUMMARY'] = True
    eval_config['OUTPUT_DETAILED'] = False # Tắt chi tiết thừa để Terminal gọn gàng hơn
    eval_config['PLOT_CURVES'] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = os.path.join(workspace, 'data', 'gt', 'mot_challenge')
    dataset_config['TRACKERS_FOLDER'] = os.path.join(workspace, 'data', 'trackers', 'mot_challenge')
    dataset_config['OUTPUT_FOLDER'] = os.path.join(workspace, 'outputs')
    dataset_config['TRACKERS_TO_EVAL'] = ['UKF_DeepSORT']
    
    # LUÔN LUÔN cấu hình là 'pedestrian' vì mã nguồn đã tráo đổi ID ở bước trên
    dataset_config['CLASSES_TO_EVAL'] = ['pedestrian'] 
    
    dataset_config['BENCHMARK'] = 'KITTI_CUSTOM'
    dataset_config['SPLIT_TO_EVAL'] = 'train'
    dataset_config['INPUT_AS_ZIP'] = False

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    # Chỉ định các bộ chỉ số đo lường cần thiết
    metrics_list = [
        trackeval.metrics.HOTA(), 
        trackeval.metrics.CLEAR(),    
        trackeval.metrics.Identity()
    ]

    try:
        evaluator.evaluate(dataset_list, metrics_list)
    except Exception as e:
        print(f"[ERROR] Quá trình chấm điểm {target_class_name.upper()} thất bại. Chi tiết lỗi: {e}")

if __name__ == '__main__':
    # Tiến hành chấm điểm riêng biệt cho Người đi bộ (MOT16 Class ID = 1)
    evaluate_single_class(target_class_name='pedestrian', target_mot_id=1)
    
    # Tiến hành chấm điểm riêng biệt cho Xe ô tô (MOT16 Class ID = 3)
    evaluate_single_class(target_class_name='car', target_mot_id=3)
    
    print("\n[SUCCESS] === HOÀN TẤT TOÀN BỘ QUÁ TRÌNH CHẤM ĐIỂM BẰNG TRACKEVAL ===")