"""
Mô đun giảm tỷ lệ khung hình (downsampling) và đổi tên lại các frame liên tục.
"""

import os
import sys
import argparse
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def downsample_frames(folder: str, rate: int = 2, reindex: bool = True) -> tuple[int, int]:
    """
    Xóa bớt frame theo tỷ lệ và đổi tên lại các frame còn lại liên tục.

    Args:
        folder (str): Thư mục chứa các tệp ảnh.
        rate (int): Tỷ lệ lấy mẫu (ví dụ: rate=2 nghĩa là cứ 2 frame giữ 1 frame, rate=3 là cứ 3 giữ 1).
        reindex (bool): Nếu True, đổi tên các frame còn lại thành số liên tục từ 1.

    Returns:
        tuple[int, int]: (số frame đã xóa, số frame còn lại)
    """
    if not os.path.isdir(folder):
        print(f"[ERROR] Thư mục không tồn tại: {folder}")
        return 0, 0

    valid_exts = (".jpg", ".jpeg", ".png")
    all_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_exts)])
    total = len(all_files)

    if total == 0:
        print(f"[WARNING] Không tìm thấy file ảnh nào trong {folder}")
        return 0, 0

    print("=" * 60)
    print(f"[INFO] Thư mục xử lý        : {folder}")
    print(f"[INFO] Tổng số frames ban đầu: {total}")
    print(f"[INFO] Tỷ lệ giảm (rate)    : {rate} (cứ {rate} frame xóa {rate - 1})")
    print(f"[INFO] Đổi tên liên tục    : {reindex}")
    print("=" * 60)

    # Chia danh sách: giữ index chia hết cho rate (0, rate, 2*rate, ...)
    kept_files = [all_files[i] for i in range(0, total, rate)]
    deleted_files = [all_files[i] for i in range(total) if i % rate != 0]

    # Bước 1: Xóa các frame không được chọn
    print(f"[INFO] Đang xóa {len(deleted_files)} frames...")
    start_time = time.time()
    for idx, fname in enumerate(deleted_files, start=1):
        fpath = os.path.join(folder, fname)
        try:
            os.remove(fpath)
        except Exception as e:
            print(f"[ERROR] Lỗi khi xóa {fname}: {e}")
        if idx % 1000 == 0 or idx == len(deleted_files):
            print(f"  Đã xóa {idx}/{len(deleted_files)} frames...", flush=True)

    # Bước 2: Đổi tên các frame còn lại nếu reindex=True
    if reindex:
        print(f"[INFO] Đang đổi tên lại {len(kept_files)} frames liên tục từ 1...")
        # Lấy prefix và số digits từ tên file đầu tiên
        sample_name = kept_files[0]
        name_no_ext, ext = os.path.splitext(sample_name)

        prefix = ""
        digits = 6
        for i, char in enumerate(name_no_ext):
            if char.isdigit():
                prefix = name_no_ext[:i]
                digits = max(6, len(name_no_ext[i:]))
                break

        for new_idx, old_filename in enumerate(kept_files, start=1):
            new_filename = f"{prefix}{new_idx:0{digits}d}{ext}"
            if old_filename != new_filename:
                old_path = os.path.join(folder, old_filename)
                new_path = os.path.join(folder, new_filename)
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f"[ERROR] Lỗi khi đổi tên {old_filename} -> {new_filename}: {e}")
            if new_idx % 1000 == 0 or new_idx == len(kept_files):
                print(f"  Đã đổi tên {new_idx}/{len(kept_files)} frames...", flush=True)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"=== HOÀN THÀNH ===")
    print(f"Số frames đã xóa  : {len(deleted_files)}")
    print(f"Số frames còn lại : {len(kept_files)}")
    print(f"Thời gian xử lý   : {elapsed:.2f}s")
    print("=" * 60, flush=True)

    return len(deleted_files), len(kept_files)


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Giảm số lượng frames và re-index lại thứ tự file liên tục.")
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="datasets/data1",
        help="Thư mục chứa các frames ảnh. Mặc định: datasets/data1"
    )
    parser.add_argument(
        "--rate", "-r",
        type=int,
        default=2,
        help="Tỷ lệ giảm: giữ 1 mỗi 'rate' frames (mặc định: 2)."
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="Không đổi tên lại các frame (giữ nguyên số thứ tự cũ)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    downsample_frames(
        folder=args.dir,
        rate=args.rate,
        reindex=not args.no_reindex
    )
