"""
Kịch bản chạy thử nghiệm luồng theo dõi đối tượng giao thông (Traffic Multi-Object Tracking Pipeline).

Tích hợp hoàn chỉnh:
  1. YOLO Detector (Mô hình YOLO26n 6 lớp giao thông: Person, Bicycle, Car, Motorbike, Bus, Truck)
  2. Unified ReID Extractor (MobileNetV4 128x128 -> Vector 256 chiều)
  3. CMC (Sparse Optical Flow bù trừ rung lắc/chuyển động camera)
  4. UKF Tracker (Unscented Kalman Filter mô hình hóa phi tuyến + Gán cặp Hungarian đa tầng)
  5. Visualizer (Hiển thị Bounding Box, quỹ đạo, nhãn ID và xuất video .mp4)

Hỗ trợ nguồn đầu vào:
  - Tệp video bất kỳ (.mp4, .avi, .mkv)
  - Webcam / Camera USB hoặc Camera CSI (0, 1)
  - Thư mục chuỗi ảnh (Image sequence folder)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataloader.dataloader import KittiParser
from src.visualizer.visualizer import Visualizer
from src.detector.detector import YOLODetector
from src.extractor.extractor import ReIDExtractor
from src.tracker.tracker import Tracker
from src.cmc.cmc import SparseOpticalFlowCMC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chạy kiểm thử hệ thống Autonomous Traffic Tracker (YOLO + ReID + UKF + CMC)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Đường dẫn file video (.mp4), webcam index ('0'), hoặc thư mục ảnh",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo26n_traffic_best.pt",
        help="Đường dẫn file trọng số YOLO PyTorch (.pt)",
    )
    parser.add_argument(
        "--reid",
        type=str,
        default="",
        help="Đường dẫn mô hình ReID (.onnx hoặc .xml). Để trống sẽ tự động dò tìm",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Ngưỡng tự tin (confidence threshold) của YOLO",
    )
    parser.add_argument(
        "--nms",
        "--iou",
        type=float,
        default=0.70,
        dest="nms",
        help="Ngưỡng NMS IoU của YOLO (mặc định 0.70 = 70%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/tracking_result.mp4",
        help="Đường dẫn lưu video kết quả đầu ra (.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Số khung hình tối đa cần xử lý (0 = xử lý toàn bộ video)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Hiển thị cửa sổ trực tiếp OpenCV khi đang xử lý",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Tỉ lệ thu nhỏ khung hình đầu vào (ví dụ 0.5 để tăng tốc)",
    )
    parser.add_argument(
        "--save-frames",
        type=str,
        default="",
        help="Đường dẫn thư mục lưu chuỗi ảnh kết quả (.jpg) sau khi tracking",
    )
    return parser.parse_args()


def resolve_input_source(input_arg):
    """Xác định nguồn video/ảnh đầu vào từ tham số hoặc tự động tìm kiếm file mẫu."""
    if input_arg:
        # Trường hợp nhập webcam index
        if input_arg.isdigit():
            return int(input_arg), "webcam"
        p = Path(input_arg)
        if p.is_file():
            return str(p), "video"
        if p.is_dir():
            if (p / "images").is_dir():
                return str(p / "images"), "dir"
            return str(p), "dir"

    # Tự động dò tìm các file video có sẵn trong project
    candidates = [
        "datasets/VID_20261020_145051.mp4",
        "outputs/videos/test_ukf_cmc.mp4",
        "datasets/KITTI_MOT/KITTI-0000",
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.isfile(c):
                return c, "video"
            if os.path.isdir(c):
                return c, "dir"

    return 0, "webcam"


def resolve_reid_model(reid_arg):
    """Tìm kiếm file mô hình ReID phù hợp nhất trong thư mục weights."""
    if reid_arg and os.path.exists(reid_arg):
        return reid_arg

    candidates = [
        "weights/unified/unified_reid.onnx",
        "weights/unified/unified_reid_int8.xml",
        "weights/unified/unified_reid_fp16.xml",
        "weights/public/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    return "weights/unified/unified_reid.onnx"


def create_frame_generator(src, src_type):
    """Tạo generator cung cấp tuần tự (frame_idx, frame, fps)."""
    if src_type in ("video", "webcam"):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở nguồn video/camera: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        def gen():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                idx += 1
                yield idx, frame, fps
            cap.release()

        return gen(), fps

    elif src_type == "dir":
        # Thư mục ảnh KITTI MOT hoặc thư mục ảnh thông thường
        if os.path.isdir(os.path.join(src, "img1")):
            parser = KittiParser(src)
            fps = 10.0

            def gen():
                for idx, frame, _ in parser.get_frame():
                    yield idx, frame, fps

            return gen(), fps
        else:
            # Thư mục ảnh thường
            valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
            files = sorted([os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(valid_exts)])
            fps = 25.0

            def gen():
                for idx, f in enumerate(files, start=1):
                    frame = cv2.imread(f)
                    if frame is not None:
                        yield idx, frame, fps

            return gen(), fps

    raise ValueError(f"Loại nguồn không hợp lệ: {src_type}")


def main():
    args = parse_args()

    input_src, src_type = resolve_input_source(args.input)
    reid_path = resolve_reid_model(args.reid)

    print("=" * 80)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG AUTONOMOUS TRAFFIC TRACKER (LAPTOP PIPELINE)")
    print("=" * 80)
    print(f"[CẤU HÌNH] Nguồn video/camera : {input_src} ({src_type})")
    print(f"[CẤU HÌNH] YOLO Detector       : {args.weights} (conf={args.conf}, nms={args.nms})")
    print(f"[CẤU HÌNH] ReID Extractor     : {reid_path}")
    print(f"[CẤU HÌNH] Lưu video kết quả  : {args.output}")
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        print(f"[CẤU HÌNH] Lưu ảnh từng frame   : {args.save_frames}")
    print(f"[CẤU HÌNH] Giới hạn số frame  : {args.max_frames if args.max_frames > 0 else 'Toàn bộ'}")
    print("=" * 80)

    # 1. Khởi tạo nguồn khung hình
    frame_gen, detected_fps = create_frame_generator(input_src, src_type)
    fps = max(1.0, min(60.0, float(detected_fps)))

    # 2. Khởi tạo Visualizer
    visualizer = Visualizer(output_path=args.output, fps=int(round(fps)))

    # 3. Khởi tạo các module mô hình AI
    print("\n[1/4] Nạp mô hình phát hiện đối tượng YOLO...")
    detector = YOLODetector(model_path=args.weights, conf_thresh=args.conf, nms_thresh=args.nms)

    print("\n[2/4] Nạp mô hình trích xuất đặc trưng ngoại hình ReID...")
    reid = ReIDExtractor(model_path=reid_path, device="CPU")

    print("\n[3/4] Khởi tạo bộ bù trừ chuyển động camera (CMC - Sparse Optical Flow)...")
    cmc_engine = SparseOpticalFlowCMC()

    print("\n[4/4] Khởi tạo lõi theo dõi Unscented Kalman Filter (UKF Tracker)...")
    tracker = Tracker(
        max_age=30,
        n_init=3,
        cosine_threshold=0.35,
        iou_threshold=0.70,
        fps=fps,
    )

    print("\n" + "-" * 80)
    print("🎬 BẮT ĐẦU XỬ LÝ VIDEO...")
    print("-" * 80)

    total_start_time = time.time()
    processed_count = 0

    try:
        for frame_idx, image, _ in frame_gen:
            if args.max_frames > 0 and processed_count >= args.max_frames:
                print(f"\n[INFO] Đã đạt giới hạn {args.max_frames} khung hình yêu cầu.")
                break

            processed_count += 1
            frame_start = time.time()

            if args.scale != 1.0:
                image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale)

            img_h, img_w = image.shape[:2]

            # Bước 1: CMC (Ước lượng chuyển động của camera)
            t_cmc = time.time()
            H_camera = cmc_engine.apply(image)
            cmc_ms = (time.time() - t_cmc) * 1000

            # Bước 2: YOLO Detection (Nhận diện 6 lớp đối tượng)
            t_yolo = time.time()
            valid_bboxes, valid_confs, valid_class_ids = detector.detect(image)
            yolo_ms = (time.time() - t_yolo) * 1000

            # Bước 3: Unified ReID (Trích xuất đặc trưng 256D)
            t_reid = time.time()
            if len(valid_bboxes) > 0:
                features = reid.extract(image, valid_bboxes)
            else:
                features = np.zeros((0, 256), dtype=np.float32)
            reid_ms = (time.time() - t_reid) * 1000

            # Bước 4: UKF Tracker Update (Dự đoán động học + Ghép cặp Hungarian)
            t_trk = time.time()
            tracker.update(
                valid_bboxes,
                valid_confs,
                valid_class_ids,
                features,
                frame_shape=(img_h, img_w),
                H_camera=H_camera,
                frame_idx=frame_idx,
            )
            trk_ms = (time.time() - t_trk) * 1000

            # Bước 5: Trực quan hóa và ghi video
            t_draw = time.time()
            draw_data = tracker.get_results()
            drawn_img = visualizer.draw_tracks(image, draw_data, frame_idx)
            draw_ms = (time.time() - t_draw) * 1000

            if args.save_frames:
                frame_out_path = os.path.join(args.save_frames, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_out_path, drawn_img)

            frame_elapsed = time.time() - frame_start
            cur_fps = 1.0 / max(1e-4, frame_elapsed)

            if processed_count % 5 == 0 or processed_count <= 10:
                print(
                    f"[FRAME {frame_idx:04d}] Objects: {len(draw_data):02d} | "
                    f"CMC: {cmc_ms:4.1f}ms | YOLO: {yolo_ms:4.1f}ms | "
                    f"ReID: {reid_ms:4.1f}ms | UKF: {trk_ms:4.1f}ms | "
                    f"FPS: {cur_fps:4.1f}"
                )

            if args.show:
                cv2.imshow("Autonomous Traffic Tracker (UKF + YOLO + ReID)", drawn_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    print("\n[INFO] Người dùng nhấn ESC để dừng.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Dừng chương trình theo tín hiệu người dùng.")

    visualizer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_elapsed = time.time() - total_start_time
    avg_fps = processed_count / max(1e-4, total_elapsed)

    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT QUÁ TRÌNH CHẠY THỬ NGHIỆM!")
    print("=" * 80)
    print(f"📊 Tổng số frame đã xử lý : {processed_count}")
    print(f"⏱️ Tổng thời gian thực thi : {total_elapsed:.2f} giây")
    print(f"🚀 Tốc độ trung bình       : {avg_fps:.2f} FPS")
    print(f"🎥 Video kết quả đã lưu tại: {args.output}")
    if args.save_frames:
        print(f"🖼️ Chuỗi ảnh đã lưu tại    : {args.save_frames}")
    print("=" * 80)


if __name__ == "__main__":
    main()
