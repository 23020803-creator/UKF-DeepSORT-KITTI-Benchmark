"""
Kịch bản đánh giá và so sánh độ chính xác chuẩn YOLO giữa:
1. Mô hình YOLO26n Gốc (Pretrained COCO, chưa huấn luyện trên tập dữ liệu Traffic).
2. Mô hình YOLO26n Đã Huấn luyện (Fine-tuned trên tập dữ liệu Traffic 6 lớp).

Chấm điểm trên 15 ảnh kiểm thử độc lập (test set) theo chuẩn mAP@0.5, mAP@0.5:0.95,
Precision, Recall và phân tích chi tiết theo từng lớp đối tượng.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def parse_imgsz(val):
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, int):
        return val
    val = str(val).strip()
    for sep in [",", "x", "X", " "]:
        if sep in val:
            parts = [int(x.strip()) for x in val.split(sep) if x.strip()]
            return parts if len(parts) > 1 else parts[0]
    return int(val)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Đánh giá & So sánh độ chính xác YOLO (Base vs Fine-tuned)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/yolo26n_dataset/data.yaml",
        help="Đường dẫn file data.yaml của tập dữ liệu",
    )
    parser.add_argument(
        "--weights-base",
        type=str,
        default="weights/yolo26n.pt",
        help="Trọng số mô hình gốc (chưa huấn luyện trên Traffic Dataset)",
    )
    parser.add_argument(
        "--weights-trained",
        type=str,
        default="",
        help="Trọng số mô hình đã huấn luyện (để trống sẽ tự động dò tìm)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Tập dữ liệu để chấm điểm (mặc định: 'test' gồm 15 ảnh)",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="384,640",
        help="Kích thước ảnh đầu vào: '384,640' (chuẩn chữ nhật 16:9) hoặc '640'",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Ngưỡng tự tin (confidence threshold) đánh giá và trực quan hóa",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="Ngưỡng NMS IoU",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Thiết bị tính toán ('cpu' hoặc '0')",
    )
    parser.add_argument(
        "--save-visual",
        action="store_true",
        default=True,
        help="Xuất ảnh so sánh trực quan bounding box giữa 2 mô hình",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation/yolo26n_benchmark",
        help="Thư mục lưu trữ báo cáo và ảnh so sánh",
    )
    return parser.parse_args()


def find_trained_weights(explicit_path=""):
    """Tự động tìm kiếm file trọng số đã huấn luyện tốt nhất."""
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)

    candidate_locations = [
        Path("weights/yolo26n_traffic_best.pt"),
        Path("weights/yolo26n_best.pt"),
        Path("weights/yolo26n_kitti_best.pt"),
        Path("weights/best.pt"),
        Path("yolo26n_best.pt"),
        Path("best.pt"),
    ]

    for p in candidate_locations:
        if p.exists() and p.stat().st_size > 1000:
            return p

    # Dò tìm trong runs/
    runs_dir = Path("runs")
    if runs_dir.exists():
        found = [p for p in runs_dir.rglob("best.pt") if p.stat().st_size > 1000]
        if found:
            return found[-1]

    return None


def evaluate_model(model_path, data_yaml, split="test", imgsz=640, device="cpu", dataset_names=None, conf=0.5):
    """Thực hiện đánh giá mô hình theo chuẩn chuẩn hóa YOLO Metrics."""
    print(f"\n[EVALUATION] Đang đánh giá mô hình: {model_path} trên tập '{split}' (conf={conf})...")
    model = YOLO(str(model_path))

    # Thực hiện val
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False,
    )

    box = metrics.box
    results = {
        "model_path": str(model_path),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "speed": metrics.speed,
        "classes": {},
    }

    # Đọc chi tiết từng lớp theo tên lớp của dataset
    ap_classes = metrics.ap_class_index if hasattr(metrics, "ap_class_index") else []
    
    for idx, c_idx in enumerate(ap_classes):
        if dataset_names and c_idx in dataset_names:
            c_name = dataset_names[c_idx]
        else:
            c_name = metrics.names.get(c_idx, f"class_{c_idx}")

        p = float(box.p[idx]) if len(box.p) > idx else 0.0
        r = float(box.r[idx]) if len(box.r) > idx else 0.0
        map50 = float(box.ap50[idx]) if len(box.ap50) > idx else 0.0
        map_all = float(box.ap[idx]) if len(box.ap) > idx else 0.0

        results["classes"][c_name] = {
            "precision": p,
            "recall": r,
            "map50": map50,
            "map50_95": map_all,
        }

    return results, model


def render_visual_comparison(base_model, trained_model, test_images, output_dir, conf=0.25, imgsz=640):
    """Vẽ ảnh so sánh trực quan dự đoán giữa 2 mô hình."""
    vis_dir = Path(output_dir) / "visual_predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[VISUAL] Đang tạo ảnh so sánh trực quan cho {len(test_images)} ảnh kiểm thử...")

    for img_path in test_images:
        img_name = Path(img_path).name
        # Dự đoán từ 2 model
        res_base = base_model.predict(img_path, conf=conf, imgsz=imgsz, verbose=False)[0]
        plot_base = res_base.plot()

        cv2.putText(
            plot_base,
            "Model Goc (Pretrained COCO)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if trained_model is not None:
            res_trained = trained_model.predict(img_path, conf=conf, imgsz=imgsz, verbose=False)[0]
            plot_trained = res_trained.plot()
            cv2.putText(
                plot_trained,
                "Model Da Huan Luyen (Traffic 6 Classes)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # Ghép 2 ảnh theo chiều dọc
            combined = np.vstack([plot_base, plot_trained])
        else:
            combined = plot_base

        out_file = vis_dir / f"compare_{img_name}"
        cv2.imwrite(str(out_file), combined)

    print(f"[SUCCESS] Đã lưu ảnh so sánh trực quan vào: {vis_dir}")


def format_delta(base_val, trained_val):
    """Tính độ chênh lệch phần trăm và giá trị tuyệt đối."""
    diff = trained_val - base_val
    diff_pct = (diff / base_val * 100) if base_val > 1e-6 else (100.0 if trained_val > 0 else 0.0)
    sign = "+" if diff >= 0 else ""
    return f"{trained_val:.4f} ({sign}{diff:.4f} | {sign}{diff_pct:.1f}%)"


def print_comparison_table(base_res, trained_res, conf=0.5):
    """In bảng so sánh chi tiết giữa 2 mô hình chuẩn Markdown/Console."""
    print("\n" + "=" * 90)
    print(f"📊 BẢNG SO SÁNH ĐỘ CHÍNH XÁC CHUẨN YOLO (10 ẢNH TEST TRAFFIC | CONF = {conf})")
    print("=" * 90)
    
    header = f"{'CHỈ SỐ ĐÁNH GIÁ':<22} | {'YOLO GỐC (COCO)':<20} | {'YOLO ĐÃ HUẤN LUYỆN':<22} | {'MỨC CẢI THIỆN (GAIN)':<20}"
    print(header)
    print("-" * 90)

    # 1. Các chỉ số tổng thể
    metrics_map = [
        ("Precision (Độ chính xác)", "precision"),
        ("Recall (Độ nhạy)", "recall"),
        ("mAP@0.5 (mAP50)", "map50"),
        ("mAP@0.5:0.95 (mAP50-95)", "map50_95"),
    ]

    for label, key in metrics_map:
        b_val = base_res[key]
        if trained_res:
            t_val = trained_res[key]
            gain_str = format_delta(b_val, t_val)
            print(f"{label:<22} | {b_val:<20.4f} | {t_val:<22.4f} | {gain_str:<20}")
        else:
            print(f"{label:<22} | {b_val:<20.4f} | {'(Chưa có weights)':<22} | {'--':<20}")

    print("-" * 90)
    # Tốc độ suy luận (Latency)
    b_speed = base_res["speed"].get("inference", 0.0)
    if trained_res:
        t_speed = trained_res["speed"].get("inference", 0.0)
        print(f"{'Inference Speed (ms)':<22} | {b_speed:<17.2f} ms | {t_speed:<19.2f} ms | {t_speed - b_speed:+.2f} ms")
    else:
        print(f"{'Inference Speed (ms)':<22} | {b_speed:<17.2f} ms | {'--':<22} | {'--':<20}")

    # 2. Chi tiết từng lớp (nếu cả 2 có thông tin)
    if trained_res and (base_res["classes"] or trained_res["classes"]):
        print("\n" + "-" * 90)
        print("🎯 CHI TIẾT mAP@0.5 THEO TỪNG LỚP ĐỐI TƯỢNG (6 TRAFFIC CLASSES):")
        print("-" * 90)
        print(f"{'LỚP ĐỐI TƯỢNG':<22} | {'mAP50 (GỐC)':<20} | {'mAP50 (ĐÃ HUẤN LUYỆN)':<22} | {'CHÊNH LỆCH':<20}")
        print("-" * 90)

        # Ưu tiên theo danh sách chuẩn 6 lớp Traffic
        traffic_order = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
        for c in traffic_order:
            b_c = base_res["classes"].get(c, {}).get("map50", 0.0)
            t_c = trained_res["classes"].get(c, {}).get("map50", 0.0)
            c_gain = format_delta(b_c, t_c)
            print(f"{c:<22} | {b_c:<20.4f} | {t_c:<22.4f} | {c_gain:<20}")

    print("=" * 90)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đọc tên lớp từ data.yaml
    data_yaml_path = Path(args.data)
    import yaml
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_names = cfg.get("names", {})
    if isinstance(dataset_names, list):
        dataset_names = {i: name for i, name in enumerate(dataset_names)}

    imgsz = parse_imgsz(args.imgsz)

    print("=" * 90)
    print("🚀 CHƯƠNG TRÌNH SO SÁNH ĐỘ CHÍNH XÁC MÔ HÌNH YOLO26N (TRAFFIC DATASET)")
    print("=" * 90)
    print(f"[CẤU HÌNH] Dataset config: {args.data}")
    print(f"[CẤU HÌNH] Split đánh giá: {args.split}")
    print(f"[CẤU HÌNH] Model gốc      : {args.weights_base}")
    print(f"[CẤU HÌNH] Kích thước ảnh: {imgsz}")

    # 1. Kiểm tra mô hình gốc
    base_weights_path = Path(args.weights_base)
    if not base_weights_path.exists():
        print(f"[ERROR] Không tìm thấy file trọng số gốc: {base_weights_path}")
        sys.exit(1)

    # 2. Đánh giá mô hình gốc
    base_results, base_model = evaluate_model(
        base_weights_path,
        args.data,
        split=args.split,
        imgsz=imgsz,
        device=args.device,
        dataset_names=dataset_names,
        conf=args.conf,
    )

    # 3. Dò tìm mô hình đã huấn luyện
    trained_weights_path = find_trained_weights(args.weights_trained)
    trained_results = None
    trained_model = None

    if trained_weights_path and trained_weights_path.exists():
        print(f"[FOUND] Tìm thấy trọng số đã huấn luyện: {trained_weights_path}")
        trained_results, trained_model = evaluate_model(
            trained_weights_path,
            args.data,
            split=args.split,
            imgsz=imgsz,
            device=args.device,
            dataset_names=dataset_names,
            conf=args.conf,
        )
    else:
        print("\n" + "!" * 90)
        print("⚠️ [THÔNG BÁO] CHƯA TÌM THẤY FILE TRỌNG SỐ ĐÃ HUẤN LUYỆN (yolo26n_traffic_best.pt / best.pt)!")
        print("!" * 90)
        print("👉 Vui lòng đặt file weights huấn luyện vào: weights/yolo26n_traffic_best.pt")
        print("   Sau đó chạy lại lệnh này để hoàn tất so sánh song song!")
        print("!" * 90)

    # 4. In bảng so sánh
    print_comparison_table(base_results, trained_results, conf=args.conf)

    # 5. Lưu báo cáo JSON
    report_file = output_dir / "comparison_summary.json"
    report_data = {
        "dataset": args.data,
        "split": args.split,
        "base_model": base_results,
        "trained_model": trained_results,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    print(f"\n[REPORT] Đã lưu báo cáo chi tiết vào: {report_file}")

    # 6. Trực quan hóa dự đoán trên tập test (10 ảnh)
    if args.save_visual:
        data_yaml_path = Path(args.data)
        import yaml
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        test_rel = cfg.get("test", "images/test")
        test_images_dir = data_yaml_path.parent / test_rel
        test_images = sorted(list(test_images_dir.glob("*.jpg")))
        
        if test_images:
            render_visual_comparison(
                base_model,
                trained_model,
                test_images,
                output_dir,
                conf=args.conf,
                imgsz=imgsz,
            )


if __name__ == "__main__":
    main()

