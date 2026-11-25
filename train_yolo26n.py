"""
Kịch bản huấn luyện (Fine-tuning) mô hình YOLO26n trên tập dữ liệu gán nhãn tùy chỉnh (6 lớp).

Hỗ trợ chạy trực tiếp trên Google Colab (GPU T4/A100) hoặc máy cục bộ (CUDA/CPU).
Sau khi huấn luyện, tự động đánh giá trên tập Validation & Test, và xuất sang
định dạng ONNX (được tối ưu hóa bằng onnxsim) và TensorRT FP16 Engine cho NVIDIA Jetson Nano.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_imgsz(val):
    """Phân tích kích thước ảnh đầu vào (hỗ trợ số nguyên hoặc tỉ lệ chữ nhật H,W)."""
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
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLO26n với Ultralytics")
    parser.add_argument(
        "--data",
        type=str,
        default="yolo26n_dataset/data.yaml",
        help="Đường dẫn đến file data.yaml của tập dữ liệu",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo26n_traffic_best.pt",
        help="Trọng số kế thừa (ưu tiên weights/yolo26n_traffic_best.pt, fallback sang weights/yolo26n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Số lượng epoch (khuyến nghị 50-100 cho tập 150 ảnh)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Kích thước batch (khuyến nghị 8 hoặc 16 trên Colab T4 GPU)",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="384,640",
        help="Kích thước ảnh đầu vào: '384,640' (chuẩn chữ nhật 16:9 cho 1080p) hoặc '640' (vuông)",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        default=True,
        help="Bật chế độ huấn luyện hình chữ nhật (Rectangular Training) để tối ưu padding",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Số epoch dừng sớm (Early stopping) nếu validation loss không cải thiện",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Số layer backbone cần đóng băng (0 = fine-tune toàn bộ, 10 = đóng băng backbone)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Thiết bị tính toán: '0' (GPU 0), 'cpu', hoặc để trống tự nhận diện",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Thư mục lưu kết quả huấn luyện",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo26n_traffic_finetune",
        help="Tên thư mục thực nghiệm",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Tự động xuất mô hình sang ONNX sau khi huấn luyện hoàn tất",
    )
    parser.add_argument(
        "--export-engine",
        action="store_true",
        default=False,
        help="Tự động xuất mô hình sang TensorRT Engine (yêu cầu GPU NVIDIA / CUDA)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    imgsz = parse_imgsz(args.imgsz)
    
    print("=" * 70)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN YOLO26N (TRAFFIC 6 CLASSES)")
    print("=" * 70)
    print(f"[CẤU HÌNH] Dataset config : {args.data}")
    print(f"[CẤU HÌNH] Pretrained weights : {args.weights}")
    print(f"[CẤU HÌNH] Epochs : {args.epochs} | Batch: {args.batch} | ImgSz: {imgsz} (Chữ nhật 16:9: {args.rect})")
    print(f"[CẤU HÌNH] Freeze layers: {args.freeze} | Patience: {args.patience}")
    print(f"[CẤU HÌNH] Project output : {args.project}/{args.name}")
    print("=" * 70)

    # 1. Kiểm tra tồn tại file config và trọng số
    data_path = Path(args.data)
    if not data_path.exists():
        # Thử tìm tương đối so với thư mục script hoặc trong datasets/
        script_dir = Path(__file__).resolve().parent
        alt_data = script_dir / args.data
        if alt_data.exists():
            data_path = alt_data
        elif (script_dir / "datasets" / args.data).exists():
            data_path = script_dir / "datasets" / args.data
        elif Path("datasets/yolo26n_dataset/data.yaml").exists():
            data_path = Path("datasets/yolo26n_dataset/data.yaml")
        else:
            raise FileNotFoundError(f"Không tìm thấy file cấu hình dữ liệu: {args.data}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / args.weights,
            script_dir / "weights" / "yolo26n_traffic_best.pt",
            Path("weights/yolo26n_traffic_best.pt"),
            script_dir / "weights" / "yolo26n.pt",
            Path("weights/yolo26n.pt"),
            Path("yolo26n.pt"),
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is not None:
            weights_path = found
        else:
            print(f"[CẢNH BÁO] Không tìm thấy '{args.weights}'. Sẽ khởi tạo từ kiến trúc yolo26n.yaml...")
            weights_path = "yolo26n.yaml"

    train_imgsz = max(imgsz) if isinstance(imgsz, (list, tuple)) else imgsz
    print(f"[INFO] Tải trọng số kế thừa từ: {weights_path}")
    model = YOLO(str(weights_path))

    # 2. Huấn luyện mô hình
    print("\n[INFO] Đang bắt đầu huấn luyện...")
    train_results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=train_imgsz,
        rect=args.rect,
        patience=args.patience,
        freeze=args.freeze if args.freeze > 0 else None,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        save=True,
        save_period=10,
        exist_ok=True,
        # Tăng cường dữ liệu phù hợp với tập 150 ảnh để tránh overfitting:
        mosaic=1.0,
        mixup=0.1,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("\n" + "=" * 70)
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("=" * 70)

    # 3. Đánh giá trên tập Validation
    print("\n[INFO] Đánh giá trên tập Validation (val)...")
    val_metrics = model.val(data=str(data_path), split="val", imgsz=train_imgsz, rect=args.rect)
    print(f"Validation mAP50:    {val_metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_metrics.box.map:.4f}")

    # 4. Đánh giá trên tập Test
    print("\n[INFO] Đánh giá trên tập Kiểm thử độc lập (test)...")
    test_metrics = model.val(data=str(data_path), split="test", imgsz=train_imgsz, rect=args.rect)
    print(f"Test mAP50:    {test_metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {test_metrics.box.map:.4f}")

    # 5. Xuất ONNX & TensorRT cho NVIDIA Jetson Nano
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(args.project) / args.name
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        import glob
        matches = glob.glob("runs/**/weights/best.pt", recursive=True)
        if matches:
            best_weights = Path(matches[-1])
    if best_weights.exists():
        print("\n" + "=" * 70)
        print("[INFO] BẮT ĐẦU XUẤT MÔ HÌNH CHO NVIDIA JETSON NANO TỪ (best.pt)...")
        print(f"[INFO] Kích thước đầu vào chuẩn chữ nhật: {imgsz}")
        print("=" * 70)
        best_model = YOLO(str(best_weights))
        
        # Xuất ONNX chuẩn hóa
        if args.export_onnx:
            try:
                print(f"[INFO] Xuất định dạng ONNX (imgsz={imgsz}, simplify=True)...")
                onnx_path = best_model.export(format="onnx", imgsz=imgsz, simplify=True)
                print(f"[SUCCESS] Xuất ONNX thành công: {onnx_path}")
                print(f"👉 Để nén sang TensorRT FP16 Engine trên Jetson Nano, chạy lệnh trtexec:")
                print(f"   /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine=yolo26n_traffic_fp16.engine --fp16\n")
            except Exception as e:
                print(f"[WARNING] Xuất ONNX thất bại: {e}")

        # Xuất TensorRT Engine trực tiếp (nếu có GPU)
        if args.export_engine:
            try:
                print(f"[INFO] Xuất định dạng TensorRT Engine (imgsz={imgsz}, FP16)...")
                engine_path = best_model.export(
                    format="engine",
                    imgsz=imgsz,
                    half=True,
                    device=args.device if args.device else 0,
                    simplify=True,
                )
                print(f"[SUCCESS] Xuất TensorRT Engine thành công: {engine_path}")
            except Exception as e:
                print(f"[WARNING] Xuất TensorRT Engine thất bại (cần NVIDIA GPU / CUDA): {e}")

    print("\n" + "=" * 70)
    print(f"🎉 TOÀN BỘ QUÁ TRÌNH ĐÃ HOÀN TẤT! Trọng số tốt nhất tại: {best_weights}")
    print("=" * 70)


if __name__ == "__main__":
    main()
