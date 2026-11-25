"""
Kịch bản xuất mô hình YOLO và ReID sang định dạng NVIDIA TensorRT Engine (FP16)
chuyên dụng cho máy tính nhúng NVIDIA Jetson Nano (Maxwell 128 cores / JetPack).

Quy trình:
1. Xuất mô hình YOLO (yolo26n_traffic_best.pt) sang TensorRT Engine FP16 trực tiếp.
2. Cung cấp lệnh trtexec để biên dịch Unified ReID sang FP16 Engine trên Jetson Nano.
"""

import os
import sys
import argparse
from pathlib import Path


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
        description="Xuất mô hình sang TensorRT FP16 cho NVIDIA Jetson Nano"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo26n_traffic_best.pt",
        help="Đường dẫn file trọng số YOLO PyTorch (.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="384,640",
        help="Kích thước ảnh đầu vào: '384,640' (chuẩn chữ nhật 16:9 cho 1080p) hoặc '640'",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Sử dụng FP16 (Half-precision) để tăng tốc tối đa trên GPU Jetson",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Thiết bị GPU để build engine (mặc định: 0)",
    )
    parser.add_argument(
        "--reid-onnx",
        type=str,
        default="weights/unified/unified_reid.onnx",
        help="Đường dẫn đến file Unified ReID ONNX",
    )
    return parser.parse_args()


def export_yolo_tensorrt(weights_path, imgsz=640, half=True, device="0"):
    """Xuất YOLO sang TensorRT Engine bằng Ultralytics."""
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Không tìm thấy file trọng số: {weights}")
        return False

    print("=" * 80)
    print("🚀 BẮT ĐẦU XUẤT MÔ HÌNH YOLO SANG TENSORRT ENGINE (FP16 CHO JETSON NANO)")
    print("=" * 80)
    print(f"[INFO] Trọng số đầu vào: {weights}")
    print(f"[INFO] Kích thước ảnh  : {imgsz}x{imgsz}")
    print(f"[INFO] Chế độ FP16     : {half}")

    try:
        from ultralytics import YOLO
        import torch

        if not torch.cuda.is_available():
            print("\n[CẢNH BÁO] Không tìm thấy NVIDIA GPU / CUDA trên máy tính hiện tại!")
            print("👉 TensorRT Engine (.engine) là định dạng phần cứng phụ thuộc chặt chẽ vào GPU.")
            print("👉 Bạn hãy thực hiện bước này trực tiếp trên máy NVIDIA Jetson Nano bằng lệnh sau:")
            print(f"\n   yolo export model={weights} format=engine half={half} imgsz={imgsz} device=0\n")
            
            # Xuất sang ONNX trước (nếu người dùng muốn dùng trtexec trên Jetson)
            print("[INFO] Đang xuất trước sang định dạng ONNX tiêu chuẩn để chuyển sang Jetson Nano...")
            model = YOLO(str(weights))
            onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
            print(f"[SUCCESS] Đã tạo file ONNX: {onnx_path}")
            print(f"👉 Trên Jetson Nano, bạn chỉ cần chạy lệnh trtexec để nén sang FP16 Engine:")
            print(f"   /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine={weights.stem}_fp16.engine --fp16")
            return True
        else:
            model = YOLO(str(weights))
            engine_path = model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                device=device,
                simplify=True,
            )
            print(f"[SUCCESS] Đã xuất thành công TensorRT Engine: {engine_path}")
            return True

    except Exception as e:
        print(f"[ERROR] Quá trình xuất TensorRT thất bại: {e}")
        return False


def guide_reid_tensorrt(reid_onnx_path):
    """Hướng dẫn biên dịch ReID ONNX sang TensorRT Engine trên Jetson Nano."""
    reid_p = Path(reid_onnx_path)
    print("\n" + "=" * 80)
    print("🚀 HƯỚNG DẪN XUẤT UNIFIED REID SANG TENSORRT FP16 (TRÊN JETSON NANO)")
    print("=" * 80)
    if reid_p.exists():
        engine_out = reid_p.with_name("unified_reid_fp16.engine")
        print(f"[INFO] File ReID ONNX đã có sẵn tại: {reid_p}")
        print("👉 Chạy lệnh sau trên terminal của Jetson Nano:")
        print(f"\n   /usr/src/tensorrt/bin/trtexec \\\n"
              f"       --onnx={reid_p} \\\n"
              f"       --saveEngine={engine_out} \\\n"
              f"       --fp16\n")
    else:
        print(f"[WARNING] Chưa tìm thấy file: {reid_p}")
    print("=" * 80)


def main():
    args = parse_args()
    imgsz = parse_imgsz(args.imgsz)
    export_yolo_tensorrt(args.weights, imgsz=imgsz, half=args.half, device=args.device)
    guide_reid_tensorrt(args.reid_onnx)


if __name__ == "__main__":
    main()

