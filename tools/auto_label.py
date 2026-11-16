"""
Mô đun tự động gán nhãn (Auto-Labeling) bằng mô hình YOLO26n (OpenVINO INT8 / PyTorch).

Công cụ này duyệt qua các thư mục datasets (từ data1 đến data10), phát hiện các đối tượng 
thuộc các lớp chỉ định (person, bicycle, car, motorbike, bus, truck) và xuất tệp nhãn .txt 
theo chuẩn định dạng YOLO (hỗ trợ huấn luyện trực tiếp và tương thích chuẩn YOLO 1.1 của CVAT).
"""

import os
import sys
import glob
import time
import argparse
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Đảm bảo mã hóa UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# Định nghĩa danh sách 6 lớp đối tượng yêu cầu theo chuẩn
TARGET_CLASSES = [
    "person",     # Target ID: 0
    "bicycle",    # Target ID: 1
    "car",        # Target ID: 2
    "motorbike",  # Target ID: 3
    "bus",        # Target ID: 4
    "truck"       # Target ID: 5
]

# Ánh xạ từ COCO 80 Class ID sang Target ID (0..5)
COCO_TO_TARGET = {
    0: 0,  # person -> person
    1: 1,  # bicycle -> bicycle
    2: 2,  # car -> car
    3: 3,  # motorcycle -> motorbike
    5: 4,  # bus -> bus
    7: 5   # truck -> truck
}


class AutoLabeler:
    """
    Trình tự động gán nhãn sử dụng mô hình YOLO26n tối ưu hóa bằng OpenVINO hoặc PyTorch.
    """

    def __init__(self, model_path: str = "weights/yolo26n_int8_openvino_model", conf_thresh: float = 0.35):
        """
        Khởi tạo AutoLabeler.

        Args:
            model_path (str): Đường dẫn tới mô hình (.pt, thư mục OpenVINO hoặc file .xml).
            conf_thresh (float): Ngưỡng độ tin cậy để lưu nhãn (mặc định: 0.35).
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.is_pt = model_path.endswith(".pt")

        if self.is_pt:
            from ultralytics import YOLO
            print(f"[INFO] Khởi tạo mô hình PyTorch Ultralytics từ: {model_path}")
            self.model = YOLO(model_path)
            self.is_openvino = False
            num_classes = len(self.model.names)
            self.is_target_model = (num_classes == len(TARGET_CLASSES))
            if self.is_target_model:
                print(f"[INFO] Phát hiện mô hình Fine-tuned {num_classes} lớp: {list(self.model.names.values())}")
            else:
                print(f"[INFO] Phát hiện mô hình Pretrained COCO {num_classes} lớp, áp dụng mapping COCO -> Target")
        else:
            import openvino as ov
            print(f"[INFO] Khởi tạo mô hình OpenVINO từ: {model_path}")
            self.core = ov.Core()

            if os.path.isfile(model_path) and model_path.endswith(".xml"):
                xml_path = model_path
            elif os.path.isdir(model_path):
                xml_candidates = glob.glob(os.path.join(model_path, "*.xml"))
                if not xml_candidates:
                    raise FileNotFoundError(f"Không tìm thấy file .xml nào trong: {model_path}")
                preferred = [f for f in xml_candidates if "yolo26" in f or "yolo11" in f]
                xml_path = preferred[0] if preferred else xml_candidates[0]
            else:
                xml_path = model_path

            model = self.core.read_model(model=xml_path)
            self.compiled_model = self.core.compile_model(model=model, device_name="CPU")
            self.output_layer = self.compiled_model.output(0)
            
            # Lấy kích thước đầu vào của model OpenVINO
            input_shape = self.compiled_model.input(0).shape
            self.net_h = int(input_shape[2])
            self.net_w = int(input_shape[3])
            self.is_openvino = True
            print(f"[SUCCESS] Đã nạp OpenVINO CPU! Input: {self.net_w}x{self.net_h}, Output shape: {self.output_layer.shape}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Phát hiện đối tượng trong ảnh và trả về danh sách bounding boxes chuẩn YOLO.

        Args:
            image (numpy.ndarray): Ảnh BGR gốc.

        Returns:
            List[Tuple[target_cls, xc, yc, w, h, conf]]: Tọa độ đã chuẩn hóa [0, 1].
        """
        h_orig, w_orig = image.shape[:2]
        labels = []

        if self.is_pt:
            predict_kwargs = {"conf": self.conf_thresh, "verbose": False}
            if not getattr(self, "is_target_model", False):
                predict_kwargs["classes"] = list(COCO_TO_TARGET.keys())

            results = self.model(image, **predict_kwargs)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    raw_cls = int(box.cls[0].cpu().numpy())
                    if getattr(self, "is_target_model", False):
                        target_cls = raw_cls
                    else:
                        if raw_cls not in COCO_TO_TARGET:
                            continue
                        target_cls = COCO_TO_TARGET[raw_cls]
                    conf = float(box.conf[0].cpu().numpy())
                    xc, yc, bw, bh = box.xywhn[0].cpu().numpy()
                    labels.append((
                        target_cls,
                        float(np.clip(xc, 0.0, 1.0)),
                        float(np.clip(yc, 0.0, 1.0)),
                        float(np.clip(bw, 0.0, 1.0)),
                        float(np.clip(bh, 0.0, 1.0)),
                        conf
                    ))
            return labels

        # Logic OpenVINO Native tối ưu hóa
        r = min(self.net_w / w_orig, self.net_h / h_orig)
        unpad_w = int(round(w_orig * r))
        unpad_h = int(round(h_orig * r))
        dw = (self.net_w - unpad_w) / 2.0
        dh = (self.net_h - unpad_h) / 2.0

        resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw_output = self.compiled_model([blob])[self.output_layer][0]

        for det in raw_output:
            x1, y1, x2, y2, conf, coco_cls = det
            if conf < self.conf_thresh:
                continue

            coco_cls_int = int(coco_cls)
            if coco_cls_int not in COCO_TO_TARGET:
                continue

            target_cls = COCO_TO_TARGET[coco_cls_int]

            # Chuyển đổi tọa độ từ không gian letterbox về không gian chuẩn hóa ảnh gốc [0, 1]
            xc = ((x1 + x2) / 2.0 - dw) / unpad_w
            yc = ((y1 + y2) / 2.0 - dh) / unpad_h
            bw = (x2 - x1) / unpad_w
            bh = (y2 - y1) / unpad_h

            xc_c = float(np.clip(xc, 0.0, 1.0))
            yc_c = float(np.clip(yc, 0.0, 1.0))
            bw_c = float(np.clip(bw, 0.0, 1.0))
            bh_c = float(np.clip(bh, 0.0, 1.0))

            if bw_c > 0 and bh_c > 0:
                labels.append((target_cls, xc_c, yc_c, bw_c, bh_c, float(conf)))

        return labels


def create_metadata_files(target_dir: str):
    """
    Tạo các tệp cấu hình (classes.txt, obj.names, data.yaml, obj.data) để tương thích 
    hoàn toàn với CVAT (định dạng YOLO 1.1) và các thư viện huấn luyện YOLO.
    """
    # 1. classes.txt & obj.names
    classes_content = "\n".join(TARGET_CLASSES) + "\n"
    with open(os.path.join(target_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write(classes_content)
    with open(os.path.join(target_dir, "obj.names"), "w", encoding="utf-8") as f:
        f.write(classes_content)

    # 2. data.yaml (Chuẩn Ultralytics YOLO)
    yaml_lines = [
        "names:",
        *[f"  {idx}: {name}" for idx, name in enumerate(TARGET_CLASSES)],
        f"nc: {len(TARGET_CLASSES)}"
    ]
    with open(os.path.join(target_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    # 3. obj.data (Chuẩn Darknet / CVAT YOLO 1.1)
    obj_data_content = (
        f"classes = {len(TARGET_CLASSES)}\n"
        "train = train.txt\n"
        "names = obj.names\n"
        "backup = backup/\n"
    )
    with open(os.path.join(target_dir, "obj.data"), "w", encoding="utf-8") as f:
        f.write(obj_data_content)


def process_dataset(
    data_dir: str,
    labeler: AutoLabeler
) -> Dict[str, int]:
    """
    Gán nhãn tự động cho một thư mục dataX.

    Args:
        data_dir (str): Đường dẫn đến thư mục dataX (chứa images/ và labels/).
        labeler (AutoLabeler): Trình gán nhãn YOLO.

    Returns:
        Dict[str, int]: Thống kê số lượng đối tượng phát hiện theo từng lớp.
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    if not os.path.isdir(images_dir):
        print(f"[WARNING] Không tìm thấy thư mục images: {images_dir}")
        return {}

    # Xóa sạch nhãn cũ nếu có để thay thế bằng tập nhãn mới
    if os.path.isdir(labels_dir):
        old_labels = glob.glob(os.path.join(labels_dir, "*.txt"))
        if old_labels:
            print(f"[INFO] Đang xóa {len(old_labels)} file nhãn cũ trong: {labels_dir}")
            for old_f in old_labels:
                try:
                    os.remove(old_f)
                except OSError:
                    pass

    os.makedirs(labels_dir, exist_ok=True)
    create_metadata_files(data_dir)

    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(valid_exts)
    ])
    total_imgs = len(image_files)

    if total_imgs == 0:
        print(f"[WARNING] Không có ảnh nào trong: {images_dir}")
        return {}

    class_stats = {cls_name: 0 for cls_name in TARGET_CLASSES}
    total_boxes = 0

    seq_name = os.path.basename(data_dir)
    print("=" * 60)
    print(f"[INFO] Bắt đầu gán nhãn cho: {seq_name} ({total_imgs} ảnh)")
    print("=" * 60)

    start_time = time.time()

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        dets = labeler.detect(img)

        # Xuất file nhãn .txt tương ứng trong thư mục labels/
        label_filename = f"{os.path.splitext(img_name)[0]}.txt"
        label_path = os.path.join(labels_dir, label_filename)

        lines = []
        for target_cls, xc, yc, bw, bh, conf in dets:
            lines.append(f"{target_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            cls_name = TARGET_CLASSES[target_cls]
            class_stats[cls_name] += 1
            total_boxes += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        if idx % 100 == 0 or idx == total_imgs:
            elapsed = time.time() - start_time
            fps = idx / elapsed if elapsed > 0 else 0
            eta = (total_imgs - idx) / fps if fps > 0 else 0
            percent = (idx / total_imgs) * 100
            print(
                f"  [{percent:5.1f}%] {seq_name}: {idx:4d}/{total_imgs} ảnh "
                f"| Tốc độ: {fps:5.1f} FPS | Đã gán: {total_boxes:5d} boxes | Còn: ~{eta:3.0f}s",
                flush=True
            )

    elapsed = time.time() - start_time
    avg_fps = total_imgs / elapsed if elapsed > 0 else 0
    print(f"[HOÀN THÀNH {seq_name}] Thời gian: {elapsed:.2f}s ({avg_fps:.1f} FPS) | Tổng boxes: {total_boxes}")
    details = ", ".join([f"{k}: {v}" for k, v in class_stats.items() if v > 0])
    print(f"  -> Chi tiết nhãn: {details if details else 'Không phát hiện đối tượng nào'}\n")

    return class_stats


def main():
    parser = argparse.ArgumentParser(description="Tự động gán nhãn bộ dữ liệu bằng YOLO26n.")
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="datasets",
        help="Thư mục gốc chứa các dataset (mặc định: datasets)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Chỉ số dataset bắt đầu (mặc định: 1)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10,
        help="Chỉ số dataset kết thúc (mặc định: 10)"
    )
    default_model = "weights/yolo26n_traffic_best.pt" if os.path.exists("weights/yolo26n_traffic_best.pt") else "weights/yolo26n_int8_openvino_model"
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=default_model,
        help=f"Đường dẫn tới mô hình (mặc định ưu tiên: {default_model})"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="Ngưỡng tự tin confidence threshold (mặc định: 0.25)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("=== PIPELINE TỰ ĐỘNG GÁN NHÃN YOLO26N CHO CVAT / HUẤN LUYỆN ===")
    print(f"Mô hình sử dụng   : {args.model}")
    print(f"Ngưỡng tự tin conf: {args.conf}")
    print(f"Các lớp gán nhãn  : {', '.join(TARGET_CLASSES)}")
    print(f"Phạm vi dataset   : data{args.start} -> data{args.end}")
    print("=" * 60)

    # Khởi tạo mô hình
    labeler = AutoLabeler(model_path=args.model, conf_thresh=args.conf)

    # Tạo tệp metadata toàn cục tại thư mục datasets/
    create_metadata_files(args.data_dir)

    grand_stats = {cls_name: 0 for cls_name in TARGET_CLASSES}
    total_datasets_processed = 0
    pipeline_start = time.time()

    for idx in range(args.start, args.end + 1):
        target_dataset_dir = os.path.join(args.data_dir, f"data{idx}")
        if not os.path.isdir(target_dataset_dir):
            print(f"[SKIP] Bỏ qua vì không tồn tại thư mục: {target_dataset_dir}")
            continue

        stats = process_dataset(target_dataset_dir, labeler)
        for k, v in stats.items():
            grand_stats[k] += v
        total_datasets_processed += 1

    total_pipeline_time = time.time() - pipeline_start
    total_all_boxes = sum(grand_stats.values())

    print("=" * 60)
    print("=== TỔNG KẾT TOÀN BỘ QUÁ TRÌNH GÁN NHÃN ===")
    print(f"Tổng số dataset đã xử lý: {total_datasets_processed}")
    print(f"Tổng thời gian thực thi : {total_pipeline_time:.2f}s (~{total_pipeline_time / 60:.1f} phút)")
    print(f"Tổng số boxes đã gán    : {total_all_boxes}")
    print("Thống kê chi tiết từng lớp:")
    for cls_idx, cls_name in enumerate(TARGET_CLASSES):
        count = grand_stats[cls_name]
        pct = (count / total_all_boxes * 100) if total_all_boxes > 0 else 0
        print(f"  [{cls_idx}] {cls_name:<10}: {count:6d} boxes ({pct:5.1f}%)")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
