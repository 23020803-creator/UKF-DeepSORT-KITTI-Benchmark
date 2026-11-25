"""
Đóng gói trọn bộ tài nguyên huấn luyện YOLO26n thành file zip cho Google Colab.
"""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ZIP = ROOT / "yolo26n_colab_package.zip"

print(f"[INFO] Bắt đầu đóng gói tài nguyên vào {PACKAGE_ZIP}...")

# Các file và thư mục cần đưa vào ZIP
dataset_dir = ROOT / "datasets" / "yolo26n_dataset"
weights_best_pt = ROOT / "weights" / "yolo26n_traffic_best.pt"
weights_pt = ROOT / "weights" / "yolo26n.pt"
weights_yaml = ROOT / "weights" / "yolo26n.yaml"
script_train = ROOT / "train_yolo26n.py"
notebook_colab = ROOT / "train_yolo26n_colab.ipynb"
readme_colab = ROOT / "README_COLAB.md"

if PACKAGE_ZIP.exists():
    PACKAGE_ZIP.unlink()

with zipfile.ZipFile(PACKAGE_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    # 1. Thêm dataset
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(ROOT / "datasets")
            zipf.write(file_path, arcname)

    # 2. Thêm weights (bao gồm bản fine-tuned để kế thừa và bản gốc làm fallback)
    if weights_best_pt.exists():
        zipf.write(weights_best_pt, Path("weights") / weights_best_pt.name)
    if weights_pt.exists():
        zipf.write(weights_pt, Path("weights") / weights_pt.name)
    if weights_yaml.exists():
        zipf.write(weights_yaml, Path("weights") / weights_yaml.name)

    # 3. Thêm script & notebook & readme
    if script_train.exists():
        zipf.write(script_train, script_train.name)
    if notebook_colab.exists():
        zipf.write(notebook_colab, notebook_colab.name)
    if readme_colab.exists():
        zipf.write(readme_colab, readme_colab.name)

# Kiểm tra tính toàn vẹn
with zipfile.ZipFile(PACKAGE_ZIP, "r") as zipf:
    test_result = zipf.testzip()
    namelist = zipf.namelist()
    print(f"[SUCCESS] Đã tạo thành công {PACKAGE_ZIP.name}")
    print(f"  - Dung lượng: {PACKAGE_ZIP.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"  - Tổng số file trong ZIP: {len(namelist)}")
    print(f"  - Kiểm tra toàn vẹn testzip(): {'OK' if test_result is None else test_result}")

    # Đếm số lượng theo thư mục
    images = [f for f in namelist if f.endswith(".jpg")]
    labels = [f for f in namelist if f.endswith(".txt") and "labels" in f]
    print(f"  - Số file ảnh: {len(images)}")
    print(f"  - Số file nhãn: {len(labels)}")

