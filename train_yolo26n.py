"""
Kịch bản huấn luyện (Fine-tuning) mô hình YOLO26n trên tập dữ liệu gán nhãn tùy chỉnh (6 lớp).

Hỗ trợ chạy trực tiếp trên Google Colab (GPU T4/A100) hoặc máy cục bộ (CUDA/CPU).
Sau khi huấn luyện, tự động đánh giá trên tập Validation & Test, và tùy chọn xuất sang
OpenVINO (FP16/INT8) và ONNX để phục vụ cho pipeline UKF-DeepSORT-KITTI-Benchmark.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO


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
        default="weights/yolo26n.pt",
        help="Trọng số khởi tạo (weights/yolo26n.pt hoặc yolo26n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số lượng epoch (khuyến nghị 50-100 cho tập 100 ảnh)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Kích thước batch (khuyến nghị 8 hoặc 16 trên Colab T4 GPU)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào (640 cho cân bằng tốc độ/độ chính xác)",
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
        "--export-openvino",
        action="store_true",
        default=True,
        help="Tự động xuất OpenVINO sau khi huấn luyện hoàn tất",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN YOLO26N (TRAFFIC 6 CLASSES)")
    print("=" * 70)
    print(f"[CẤU HÌNH] Dataset config : {args.data}")
    print(f"[CẤU HÌNH] Pretrained weights : {args.weights}")
    print(f"[CẤU HÌNH] Epochs : {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
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
        alt_weights = script_dir / args.weights
        if alt_weights.exists():
            weights_path = alt_weights
        elif (script_dir / "weights" / "yolo26n.pt").exists():
            weights_path = script_dir / "weights" / "yolo26n.pt"
        elif Path("weights/yolo26n.pt").exists():
            weights_path = Path("weights/yolo26n.pt")
        elif Path("yolo26n.pt").exists():
            weights_path = Path("yolo26n.pt")
        else:
            print(f"[CẢNH BÁO] Không tìm thấy '{args.weights}'. Sẽ khởi tạo từ kiến trúc yolo26n.yaml...")
            weights_path = "yolo26n.yaml"

    print(f"[INFO] Tải mô hình từ: {weights_path}")
    model = YOLO(str(weights_path))

    # 2. Huấn luyện mô hình
    print("\n[INFO] Đang bắt đầu huấn luyện...")
    train_results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        freeze=args.freeze if args.freeze > 0 else None,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        save=True,
        save_period=10,
        exist_ok=True,
        # Tăng cường dữ liệu phù hợp với tập 100 ảnh để tránh overfitting:
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
    val_metrics = model.val(data=str(data_path), split="val")
    print(f"Validation mAP50:    {val_metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_metrics.box.map:.4f}")

    # 4. Đánh giá trên tập Test
    print("\n[INFO] Đánh giá trên tập Kiểm thử độc lập (test)...")
    test_metrics = model.val(data=str(data_path), split="test")
    print(f"Test mAP50:    {test_metrics.box.map50:.4f}")
    print(f"Test mAP50-95: {test_metrics.box.map:.4f}")

    # 5. Xuất OpenVINO & ONNX (nếu được yêu cầu)
    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(args.project) / args.name
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        import glob
        matches = glob.glob("runs/**/weights/best.pt", recursive=True)
        if matches:
            best_weights = Path(matches[-1])
    if best_weights.exists() and args.export_openvino:
        print("\n" + "=" * 70)
        print("[INFO] BẮT ĐẦU XUẤT MÔ HÌNH SANG OPENVINO & ONNX TỪ TRỌNG SỐ TỐT NHẤT (best.pt)...")
        print("=" * 70)
        best_model = YOLO(str(best_weights))
        
        # Xuất ONNX
        try:
            print("[INFO] Xuất định dạng ONNX...")
            best_model.export(format="onnx", imgsz=args.imgsz)
            print("[SUCCESS] Xuất ONNX thành công!")
        except Exception as e:
            print(f"[WARNING] Xuất ONNX thất bại: {e}")

        # Xuất OpenVINO FP16
        try:
            print("[INFO] Xuất định dạng OpenVINO FP16...")
            best_model.export(format="openvino", imgsz=args.imgsz, half=True)
            print("[SUCCESS] Xuất OpenVINO FP16 thành công!")
        except Exception as e:
            print(f"[WARNING] Xuất OpenVINO FP16 thất bại: {e}")

        # Xuất OpenVINO INT8 (với Calibration data)
        try:
            print("[INFO] Xuất định dạng OpenVINO INT8 (Post-Training Quantization)...")
            best_model.export(
                format="openvino",
                int8=True,
                data=str(data_path),
                imgsz=args.imgsz,
            )
            print("[SUCCESS] Xuất OpenVINO INT8 thành công!")
        except Exception as e:
            print(f"[WARNING] Xuất OpenVINO INT8 thất bại: {e}")

    print("\n" + "=" * 70)
    print(f"🎉 TOÀN BỘ QUÁ TRÌNH ĐÃ HOÀN TẤT! Trọng số tốt nhất tại: {best_weights}")
    print("=" * 70)


if __name__ == "__main__":
    main()
