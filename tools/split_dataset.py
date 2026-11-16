"""
Mô đun chia tập dữ liệu thành N thư mục (data1..dataN) theo chuẩn cấu trúc YOLO/Tracking:
dataX/
  ├── images/
  └── labels/
"""

import os
import sys
import shutil
import argparse
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def split_dataset(
    source_dir: str = "datasets/data1",
    base_dir: str = "datasets",
    num_splits: int = 10,
    prefix_name: str = "data",
    reindex: bool = True
):
    """
    Chia đều các ảnh từ source_dir thành num_splits thư mục con với cấu trúc images/ và labels/.

    Args:
        source_dir (str): Thư mục chứa toàn bộ ảnh ban đầu.
        base_dir (str): Thư mục gốc chứa các thư mục dataX đầu ra.
        num_splits (int): Số lượng thư mục cần chia (mặc định: 10).
        prefix_name (str): Tiền tố cho tên thư mục đầu ra (mặc định: 'data' -> data1, data2...).
        reindex (bool): Đánh số lại từ frame_000001.jpg trong mỗi thư mục.
    """
    if not os.path.isdir(source_dir):
        print(f"[ERROR] Thư mục nguồn không tồn tại: {source_dir}")
        return

    # Lấy danh sách ảnh hợp lệ
    valid_exts = (".jpg", ".jpeg", ".png")
    all_files = sorted([
        f for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith(valid_exts)
    ])
    total = len(all_files)

    if total == 0:
        print(f"[ERROR] Không tìm thấy file ảnh nào trong {source_dir}")
        return

    print("=" * 60)
    print(f"[INFO] Thư mục nguồn      : {source_dir}")
    print(f"[INFO] Tổng số ảnh        : {total}")
    print(f"[INFO] Số lượng thư mục   : {num_splits} ({prefix_name}1 -> {prefix_name}{num_splits})")
    print(f"[INFO] Cấu trúc thư mục   : {prefix_name}X/images và {prefix_name}X/labels")
    print(f"[INFO] Đánh số lại từ đầu : {reindex}")
    print("=" * 60)

    # Tính toán số lượng ảnh cho mỗi split
    base_count = total // num_splits
    remainder = total % num_splits

    counts = [base_count + (1 if i < remainder else 0) for i in range(num_splits)]
    print(f"[INFO] Phân bổ số ảnh: {counts}")

    start_time = time.time()
    file_ptr = 0

    # Tạo trước toàn bộ thư mục images/ và labels/
    for i in range(1, num_splits + 1):
        target_dir = os.path.join(base_dir, f"{prefix_name}{i}")
        images_dir = os.path.join(target_dir, "images")
        labels_dir = os.path.join(target_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    # Tiến hành di chuyển và đổi tên
    for split_idx, count in enumerate(counts, start=1):
        target_images_dir = os.path.join(base_dir, f"{prefix_name}{split_idx}", "images")
        split_files = all_files[file_ptr:file_ptr + count]
        file_ptr += count

        for new_idx, old_filename in enumerate(split_files, start=1):
            src_path = os.path.join(source_dir, old_filename)
            ext = os.path.splitext(old_filename)[1]
            if reindex:
                new_filename = f"frame_{new_idx:06d}{ext}"
            else:
                new_filename = old_filename
            dst_path = os.path.join(target_images_dir, new_filename)

            shutil.move(src_path, dst_path)

        first_sample = f"frame_000001{ext}" if reindex else split_files[0]
        last_sample = f"frame_{count:06d}{ext}" if reindex else split_files[-1]
        print(
            f"  [ĐÃ XONG] {prefix_name}{split_idx}: {count} ảnh "
            f"({first_sample} -> {last_sample})",
            flush=True
        )

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"=== HOÀN THÀNH CHIA DATASET ===")
    print(f"Tổng số ảnh đã chia : {total}")
    print(f"Thời gian thực hiện : {elapsed:.2f}s")
    for i in range(1, num_splits + 1):
        p = os.path.join(base_dir, f"{prefix_name}{i}")
        num_imgs = len(os.listdir(os.path.join(p, "images")))
        print(f"  {prefix_name}{i}/: {num_imgs} images, labels/ (rỗng)")
    print("=" * 60, flush=True)


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chia dataset thành các thư mục con images/ và labels/.")
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="datasets/data1",
        help="Thư mục chứa ảnh nguồn (mặc định: datasets/data1)"
    )
    parser.add_argument(
        "--base", "-b",
        type=str,
        default="datasets",
        help="Thư mục chứa các thư mục dataX (mặc định: datasets)"
    )
    parser.add_argument(
        "--splits", "-n",
        type=int,
        default=10,
        help="Số lượng tập cần chia (mặc định: 10)"
    )
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        default="data",
        help="Tiền tố thư mục (mặc định: data -> data1..data10)"
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Không đánh số lại từ đầu trong từng thư mục"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        source_dir=args.source,
        base_dir=args.base,
        num_splits=args.splits,
        prefix_name=args.prefix,
        reindex=not args.no_reindex
    )
