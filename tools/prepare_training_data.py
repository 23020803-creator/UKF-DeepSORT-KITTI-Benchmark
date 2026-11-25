"""
Script chuẩn bị và phân chia tập dữ liệu 150 ảnh đã gán nhãn lại từ CVAT:
1. Sao lưu nhãn cũ và cập nhật nhãn mới vào annotations/cvat/ và datasets/dataX/labels/
2. Xóa dữ liệu cũ và phân chia 150 ảnh thành Train (70%), Val (20%), Test (10%)
3. Tạo data.yaml và split_summary.json chuẩn cho Ultralytics YOLO26n
"""

import os
import sys
import shutil
import zipfile
import csv
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# Thiết lập bảng mã UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
ZIP_NEW_LABELS = ROOT / "job_4659740_annotations_2026_11_24_15_01_54_yolo 1.1.zip"
CVAT_DIR = ROOT / "annotations" / "cvat" / "cvat_random150_conf010_20261027"
IMAGE_POOL_DIR = CVAT_DIR / "subset" / "data1" / "images"
DATASET_OUT = ROOT / "datasets" / "yolo26n_dataset"

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "bus",
    "truck"
]


def step1_sync_labels():
    print("=" * 70)
    print("BƯỚC 1: ĐỒNG BỘ VÀ CẬP NHẬT NHÃN MỚI TỪ CVAT")
    print("=" * 70)

    if not ZIP_NEW_LABELS.exists():
        raise FileNotFoundError(f"Không tìm thấy file zip nhãn mới: {ZIP_NEW_LABELS}")

    # 1.1 Sao lưu ZIP cũ
    old_zip = CVAT_DIR / "150_annotations_yolo1.1.zip"
    backup_zip = CVAT_DIR / "150_annotations_yolo1.1_old_autolabel.zip"
    if old_zip.exists() and not backup_zip.exists():
        shutil.copy2(old_zip, backup_zip)
        print(f"[OK] Đã sao lưu nhãn cũ thành: {backup_zip.name}")

    # Copy ZIP mới thành 150_annotations_yolo1.1.zip
    shutil.copy2(ZIP_NEW_LABELS, old_zip)
    print(f"[OK] Đã cập nhật tệp nhãn chính thức: {old_zip.name}")

    # 1.2 Giải nén nhãn mới vào cvat_ready/labels
    cvat_labels_dir = CVAT_DIR / "cvat_ready" / "labels"
    if cvat_labels_dir.exists():
        shutil.rmtree(cvat_labels_dir)
    cvat_labels_dir.mkdir(parents=True, exist_ok=True)

    extracted_labels = {}
    with zipfile.ZipFile(ZIP_NEW_LABELS, "r") as z:
        for member in z.namelist():
            if member.startswith("obj_train_data/") and member.endswith(".txt"):
                fname = Path(member).name
                content = z.read(member).decode("utf-8")
                (cvat_labels_dir / fname).write_text(content, encoding="utf-8")
                extracted_labels[Path(fname).stem] = content

    print(f"[OK] Đã xuất {len(extracted_labels)} file nhãn mới vào: {cvat_labels_dir}")

    # 1.3 Cập nhật vào datasets/data1..data10/labels
    selection_file = CVAT_DIR / "selection.csv"
    updated_sources = 0
    if selection_file.exists():
        with open(selection_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_img = row["source"]
                cvat_filename = row["filename"]
                stem = Path(cvat_filename).stem
                src_stem = Path(source_img).stem

                src_label = ROOT / Path(source_img).parent.parent / "labels" / f"{src_stem}.txt"
                if stem in extracted_labels:
                    src_label.parent.mkdir(parents=True, exist_ok=True)
                    src_label.write_text(extracted_labels[stem], encoding="utf-8")
                    updated_sources += 1

    print(f"[OK] Đã đồng bộ {updated_sources} file nhãn chuẩn vào datasets/data1..data10/labels/")

    # Dọn dẹp file tạm ở root nếu có
    for temp_f in [ROOT / "obj.data", ROOT / "obj.names"]:
        if temp_f.exists():
            temp_f.unlink()
            print(f"[OK] Đã xóa file tạm: {temp_f.name}")


def step2_split_dataset():
    print("\n" + "=" * 70)
    print("BƯỚC 2: PHÂN CHIA TẬP DỮ LIỆU 150 ẢNH (TRAIN: 70% | VAL: 20% | TEST: 10%)")
    print("=" * 70)

    # Đọc danh sách 150 ảnh và nhãn
    cvat_labels_dir = CVAT_DIR / "cvat_ready" / "labels"
    label_files = sorted(list(cvat_labels_dir.glob("*.txt")))
    assert len(label_files) == 150, f"Dự kiến 150 file nhãn nhưng có {len(label_files)}"

    # Phân nhóm theo video
    video_groups = defaultdict(list)
    for lf in label_files:
        stem = lf.stem
        # Tên có dạng data01_frame_000072
        video_prefix = stem.split("_")[0]
        
        # Đọc nội dung để biết có lớp nào
        content = lf.read_text(encoding="utf-8").strip()
        classes_in_file = set()
        box_count = 0
        class_dist = Counter()
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cid = int(parts[0])
            classes_in_file.add(cid)
            class_dist[cid] += 1
            box_count += 1

        video_groups[video_prefix].append({
            "stem": stem,
            "label_file": lf,
            "img_file": IMAGE_POOL_DIR / f"{stem}.jpg",
            "video": video_prefix,
            "classes": classes_in_file,
            "boxes": box_count,
            "class_dist": dict(class_dist)
        })

    print(f"[INFO] Phân bổ theo video: 10 video, mỗi video {len(next(iter(video_groups.values())))} ảnh.")

    # Phân bổ mục tiêu:
    # Tổng: 150 ảnh -> Train: 105 (70%), Val: 30 (20%), Test: 15 (10%)
    # Mỗi video có 15 ảnh.
    # 5 video: 11 Train, 3 Val, 1 Test
    # 5 video: 10 Train, 3 Val, 2 Test
    # Tổng = 55 + 50 = 105 Train; 10 * 3 = 30 Val; 5 * 1 + 5 * 2 = 15 Test.
    # Đồng thời đảm bảo lớp hiếm 'bicycle' (cid=1) có mặt ở cả Train, Val, Test.

    # Các ảnh có bicycle:
    # data02: data02_frame_000006, data02_frame_000130, data02_frame_000476
    # data09: data09_frame_000103, data09_frame_000109

    splits_data = {"train": [], "val": [], "test": []}

    # Đặt seed cố định để tái lập được kết quả
    rng = random.Random(42)

    # Xác định số lượng test cho từng video (5 video có 2 test, 5 video có 1 test)
    # Ta ưu tiên data02 có 1 test, data09 có 1 test để chia bicycle
    test_counts = {
        "data01": 2,
        "data02": 1,
        "data03": 2,
        "data04": 2,
        "data05": 1,
        "data06": 2,
        "data07": 1,
        "data08": 2,
        "data09": 1,
        "data10": 1,
    }
    # Tổng test = 2 + 1 + 2 + 2 + 1 + 2 + 1 + 2 + 1 + 1 = 15
    # Val mỗi video = 3 -> Tổng val = 30
    # Train mỗi video = 15 - 3 - test_count -> Tổng train = 105

    for v_prefix in sorted(video_groups.keys()):
        items = list(video_groups[v_prefix])
        v_test_cnt = test_counts[v_prefix]
        v_val_cnt = 3
        v_train_cnt = 15 - v_val_cnt - v_test_cnt

        if v_prefix == "data02":
            # 3 bicycle items: 2 Train, 1 Val, 0 Test (since v_test_cnt=1, we can put 1 bicycle in Test or Val)
            # data02 has v_train_cnt=11, v_val_cnt=3, v_test_cnt=1
            b_items = [it for it in items if 1 in it["classes"]]
            nb_items = [it for it in items if 1 not in it["classes"]]
            rng.shuffle(b_items)
            rng.shuffle(nb_items)
            # 2 bicycle to train, 1 bicycle to val
            train_items = b_items[:2] + nb_items[:(v_train_cnt - 2)]
            val_items = [b_items[2]] + nb_items[(v_train_cnt - 2):(v_train_cnt - 2 + v_val_cnt - 1)]
            test_items = nb_items[(v_train_cnt - 2 + v_val_cnt - 1):]
        elif v_prefix == "data09":
            # 2 bicycle items: 1 Train, 0 Val, 1 Test
            # data09 has v_train_cnt=11, v_val_cnt=3, v_test_cnt=1
            b_items = [it for it in items if 1 in it["classes"]]
            nb_items = [it for it in items if 1 not in it["classes"]]
            rng.shuffle(b_items)
            rng.shuffle(nb_items)
            # 1 bicycle to train, 1 bicycle to test
            train_items = [b_items[0]] + nb_items[:(v_train_cnt - 1)]
            val_items = nb_items[(v_train_cnt - 1):(v_train_cnt - 1 + v_val_cnt)]
            test_items = [b_items[1]] + nb_items[(v_train_cnt - 1 + v_val_cnt):(v_train_cnt - 1 + v_val_cnt + v_test_cnt - 1)]
        else:
            rng.shuffle(items)
            train_items = items[:v_train_cnt]
            val_items = items[v_train_cnt:v_train_cnt + v_val_cnt]
            test_items = items[v_train_cnt + v_val_cnt:]

        assert len(train_items) == v_train_cnt, f"Sai kích thước train tại {v_prefix}"
        assert len(val_items) == v_val_cnt, f"Sai kích thước val tại {v_prefix}"
        assert len(test_items) == v_test_cnt, f"Sai kích thước test tại {v_prefix}"

        splits_data["train"].extend(train_items)
        splits_data["val"].extend(val_items)
        splits_data["test"].extend(test_items)

    print(f"[OK] Phân chia hoàn tất:")
    print(f"  - Train : {len(splits_data['train'])} ảnh (70%)")
    print(f"  - Val   : {len(splits_data['val'])} ảnh (20%)")
    print(f"  - Test  : {len(splits_data['test'])} ảnh (10%)")

    # Xóa thư mục yolo26n_dataset cũ để thay mới hoàn toàn
    if DATASET_OUT.exists():
        shutil.rmtree(DATASET_OUT)

    for split in ["train", "val", "test"]:
        (DATASET_OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    records = []
    split_class_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }

    for split, items in splits_data.items():
        for item in items:
            stem = item["stem"]
            src_img = item["img_file"]
            src_lbl = item["label_file"]

            dst_img = DATASET_OUT / "images" / split / f"{stem}.jpg"
            dst_lbl = DATASET_OUT / "labels" / split / f"{stem}.txt"

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

            for cid, cnt in item["class_dist"].items():
                split_class_counts[split][cid] += cnt

            records.append({
                "stem": stem,
                "split": split,
                "video": item["video"],
                "boxes": item["boxes"],
                "class_distribution": item["class_dist"]
            })

    # Tạo data.yaml
    data_yaml_content = (
        "# YOLO Dataset Configuration for KITTI Benchmark Fine-Tuning\n"
        "# 150 Frames - 6 Classes Traffic Tracking\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "# Classes (6 classes defined in KITTI tracking benchmark)\n"
        "names:\n"
        "  0: person\n"
        "  1: bicycle\n"
        "  2: car\n"
        "  3: motorbike\n"
        "  4: bus\n"
        "  5: truck\n"
        "nc: 6\n"
    )
    (DATASET_OUT / "data.yaml").write_text(data_yaml_content, encoding="utf-8")
    print(f"[OK] Đã tạo file cấu hình: {DATASET_OUT / 'data.yaml'}")

    # Tạo split_summary.json
    summary = {
        "total_images": 150,
        "split_ratio": "70/20/10",
        "train_count": len(splits_data["train"]),
        "val_count": len(splits_data["val"]),
        "test_count": len(splits_data["test"]),
        "total_boxes": sum(r["boxes"] for r in records),
        "class_distribution": {
            "train": {CLASS_NAMES[c]: split_class_counts["train"][c] for c in range(6)},
            "val": {CLASS_NAMES[c]: split_class_counts["val"][c] for c in range(6)},
            "test": {CLASS_NAMES[c]: split_class_counts["test"][c] for c in range(6)},
        },
        "records": records
    }
    with open(DATASET_OUT / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Đã tạo file thống kê chi tiết: {DATASET_OUT / 'split_summary.json'}")

    print("\n" + "=" * 70)
    print("THỐNG KÊ PHÂN BỐ LỚP THEO TẬP:")
    print("=" * 70)
    print(f"{'Lớp đối tượng':<15} | {'Train (105)':<12} | {'Val (30)':<10} | {'Test (15)':<10} | {'Tổng (150)':<10}")
    print("-" * 65)
    for c in range(6):
        cname = CLASS_NAMES[c]
        tr = split_class_counts["train"][c]
        va = split_class_counts["val"][c]
        te = split_class_counts["test"][c]
        tot = tr + va + te
        print(f"{c} - {cname:<11} | {tr:<12d} | {va:<10d} | {te:<10d} | {tot:<10d}")
    print("-" * 65)
    total_tr = sum(split_class_counts["train"].values())
    total_va = sum(split_class_counts["val"].values())
    total_te = sum(split_class_counts["test"].values())
    print(f"{'TỔNG BOXES':<15} | {total_tr:<12d} | {total_va:<10d} | {total_te:<10d} | {total_tr + total_va + total_te:<10d}")


if __name__ == "__main__":
    step1_sync_labels()
    step2_split_dataset()
