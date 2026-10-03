"""
Kịch bản chuyển đổi bộ dữ liệu KITTI gốc sang định dạng chuẩn MOT Challenge.

Bộ dữ liệu KITTI Tracking gốc có cấu trúc:
- image_02/<seq>/<000000.png ...>
- label_02/<seq>.txt (định dạng: frame track_id type truncated occluded alpha x1 y1 x2 y2 ...)

Kịch bản này chuyển đổi sang định dạng MOT16:
- datasets/KITTI_MOT/KITTI-<seq>/img1/<000001.jpg ...>
- datasets/KITTI_MOT/KITTI-<seq>/gt/gt.txt (<frame>, <id>, <x>, <y>, <w>, <h>, 1, <mot_class>, 1)
- datasets/KITTI_MOT/KITTI-<seq>/seqinfo.ini
"""

import os
import shutil
import glob

# Bảng ánh xạ lớp KITTI thô sang MOT16 Class ID
KITTI_TYPE_TO_MOT = {
    'Pedestrian': 1,
    'Person_sitting': 2,
    'Car': 3,
    'Van': 3,
    'Truck': 3,
    'Cyclist': 4,
    'DontCare': 12,
    'Misc': 12
}

def convert_kitti_sequence(kitti_base_dir, seq_idx_str, output_base_dir):
    """
    Chuyển đổi một chuỗi video từ KITTI gốc sang chuẩn MOT.
    """
    label_file = os.path.join(kitti_base_dir, "training", "label_02", f"{seq_idx_str}.txt")
    img_dir = os.path.join(kitti_base_dir, "training", "image_02", seq_idx_str)

    if not os.path.exists(label_file) or not os.path.exists(img_dir):
        print(f"[SKIP] Bỏ qua {seq_idx_str}: Không tìm thấy thư mục ảnh hoặc nhãn.")
        return

    mot_seq_name = f"KITTI-{seq_idx_str}"
    target_seq_dir = os.path.join(output_base_dir, mot_seq_name)
    target_img_dir = os.path.join(target_seq_dir, "img1")
    target_gt_dir = os.path.join(target_seq_dir, "gt")

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)

    # 1. Chuyển đổi nhãn label_02 sang gt.txt
    gt_lines = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            frame_kitti = int(parts[0])
            track_id = int(parts[1])
            obj_type = parts[2]
            
            # Bỏ qua nếu là loại không xác định
            mot_class = KITTI_TYPE_TO_MOT.get(obj_type, None)
            if mot_class is None:
                continue

            # KITTI 0-indexed frame -> MOT 1-indexed frame
            frame_mot = frame_kitti + 1

            x1 = float(parts[6])
            y1 = float(parts[7])
            x2 = float(parts[8])
            y2 = float(parts[9])

            w = x2 - x1
            h = y2 - y1

            # Format MOT16: frame, id, x, y, w, h, mark=1, class, visibility=1
            gt_line = f"{frame_mot},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{mot_class},1\n"
            gt_lines.append(gt_line)

    gt_out_path = os.path.join(target_gt_dir, "gt.txt")
    with open(gt_out_path, "w") as f:
        f.writelines(gt_lines)

    # 2. Tạo seqinfo.ini
    raw_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
    seq_length = len(raw_imgs)
    
    seqinfo_path = os.path.join(target_seq_dir, "seqinfo.ini")
    with open(seqinfo_path, "w") as f:
        f.write(f"[Sequence]\n")
        f.write(f"name={mot_seq_name}\n")
        f.write(f"imDir=img1\n")
        f.write(f"frameRate=10\n")
        f.write(f"seqLength={seq_length}\n")
        f.write(f"imWidth=1242\n")
        f.write(f"imHeight=375\n")
        f.write(f"imExt=.jpg\n")

    print(f"[SUCCESS] Đã tạo cấu trúc MOT cho {mot_seq_name} ({len(gt_lines)} nhãn, {seq_length} frames).")

def main():
    kitti_base = "datasets/KITTI"
    mot_output = "datasets/KITTI_MOT"

    if not os.path.exists(kitti_base):
        print(f"[ERROR] Không tìm thấy thư mục gốc KITTI: {kitti_base}")
        return

    label_files = sorted(glob.glob(os.path.join(kitti_base, "training", "label_02", "*.txt")))
    print(f"[INFO] Tìm thấy {len(label_files)} chuỗi dữ liệu trong {kitti_base}.")

    for lf in label_files:
        seq_idx_str = os.path.splitext(os.path.basename(lf))[0]
        convert_kitti_sequence(kitti_base, seq_idx_str, mot_output)

if __name__ == "__main__":
    main()