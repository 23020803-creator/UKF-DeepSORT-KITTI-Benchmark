"""
Mô đun đóng gói nhãn sang định dạng chuẩn CVAT YOLO 1.1 ZIP.

CVAT yêu cầu cấu trúc tệp nén YOLO 1.1 nghiêm ngặt:
archive.zip/
  ├── obj.data
  ├── obj.names
  ├── train.txt
  └── obj_train_data/
        ├── frame_000001.txt
        ├── frame_000002.txt
        └── ...
"""

import os
import sys
import zipfile
import argparse
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


TARGET_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "truck"
]


def create_cvat_zip(data_dir: str, output_zip_path: str):
    """
    Đóng gói nhãn của một thư mục dataX thành tệp zip chuẩn CVAT YOLO 1.1.
    """
    labels_dir = os.path.join(data_dir, "labels")
    if not os.path.isdir(labels_dir):
        print(f"[WARNING] Không tìm thấy thư mục nhãn: {labels_dir}")
        return False

    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    if not label_files:
        print(f"[WARNING] Không có file nhãn .txt nào trong: {labels_dir}")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)

    # 1. Tạo nội dung obj.names
    obj_names_content = "\n".join(TARGET_CLASSES) + "\n"

    # 2. Tạo nội dung train.txt
    train_lines = [f"data/obj_train_data/{os.path.splitext(f)[0]}.jpg\n" for f in label_files]
    train_content = "".join(train_lines)

    # 3. Tạo nội dung obj.data
    obj_data_content = (
        f"classes = {len(TARGET_CLASSES)}\n"
        "train = train.txt\n"
        "names = obj.names\n"
        "backup = backup/\n"
    )

    # 4. Ghi vào tệp ZIP
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("obj.names", obj_names_content)
        zipf.writestr("obj.data", obj_data_content)
        zipf.writestr("train.txt", train_content)

        for fname in label_files:
            fpath = os.path.join(labels_dir, fname)
            arcname = f"obj_train_data/{fname}"
            zipf.write(fpath, arcname)

    return True


def package_all(data_base_dir: str = "datasets", out_dir: str = "annotations/cvat/exports", start: int = 1, end: int = 10):
    """
    Đóng gói toàn bộ các dataset từ data{start} đến data{end}.
    """
    print("=" * 60)
    print("=== ĐÓNG GÓI NHÃN CHUẨN CVAT YOLO 1.1 ZIP ===")
    print(f"Thư mục nguồn : {data_base_dir}")
    print(f"Thư mục xuất  : {out_dir}")
    print(f"Phạm vi       : data{start} -> data{end}")
    print("=" * 60)

    start_time = time.time()
    success_count = 0

    for idx in range(start, end + 1):
        data_dir = os.path.join(data_base_dir, f"data{idx}")
        if not os.path.isdir(data_dir):
            continue

        zip_name = f"data{idx}_cvat_yolo1.1.zip"
        zip_path = os.path.join(out_dir, zip_name)

        if create_cvat_zip(data_dir, zip_path):
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            num_labels = len([f for f in os.listdir(os.path.join(data_dir, "labels")) if f.endswith(".txt")])
            print(f"  [XONG] {zip_name:<24} | {num_labels:4d} files | Kích thước: {size_mb:.2f} MB")
            success_count += 1

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Đã tạo thành công {success_count} file zip trong thư mục: {os.path.abspath(out_dir)}")
    print(f"Thời gian: {elapsed:.2f}s")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đóng gói nhãn sang ZIP chuẩn CVAT YOLO 1.1.")
    parser.add_argument("--base", default="datasets", help="Thư mục datasets")
    parser.add_argument("--out", default="annotations/cvat/exports", help="Thư mục lưu các file zip (mặc định: annotations/cvat/exports)")
    parser.add_argument("--start", type=int, default=1, help="Bắt đầu từ dataX")
    parser.add_argument("--end", type=int, default=10, help="Kết thúc tại dataX")
    args = parser.parse_args()

    package_all(data_base_dir=args.base, out_dir=args.out, start=args.start, end=args.end)
