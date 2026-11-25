"""
Mô đun YOLO Detector hỗ trợ đầy đủ 6 lớp đối tượng giao thông (Traffic Multi-Class Tracking).

Hỗ trợ nạp trực tiếp mô hình PyTorch (.pt) hoặc TensorRT Engine (.engine) cho Jetson Nano.
Định dạng 6 lớp đối tượng giao thông chuẩn hóa:
  0: Person (Người đi bộ)
  1: Bicycle (Xe đạp / Người đi xe đạp)
  2: Car (Xe ô tô con)
  3: Motorbike (Xe máy / Người lái xe máy)
  4: Bus (Xe buýt)
  5: Truck (Xe tải)
"""

import os
import glob
import cv2
import numpy as np
import torch
from pathlib import Path


class YOLODetector:
    """
    Lớp đóng gói mô hình YOLO Detector hỗ trợ 6 lớp giao thông.

    Attributes:
        conf_thresh (float): Ngưỡng tự tin tối thiểu để giữ lại Bounding Box.
        img_width (int): Chiều rộng ảnh chuẩn hóa đầu vào của mô hình.
        img_height (int): Chiều cao ảnh chuẩn hóa đầu vào của mô hình.
        classes_map (dict): Tên ánh xạ 6 lớp đối tượng giao thông.
    """

    # Bảng 6 lớp đối tượng chuẩn của hệ thống giám sát giao thông
    TRAFFIC_CLASSES = {
        0: "Person",
        1: "Bicycle",
        2: "Car",
        3: "Motorbike",
        4: "Bus",
        5: "Truck",
    }

    # Ánh xạ từ COCO 80 classes sang 6 classes Traffic (nếu dùng model COCO gốc chưa fine-tune)
    COCO_TO_TRAFFIC = {
        0: 0,  # person -> Person
        1: 1,  # bicycle -> Bicycle
        2: 2,  # car -> Car
        3: 3,  # motorcycle -> Motorbike
        5: 4,  # bus -> Bus
        7: 5,  # truck -> Truck
    }

    def __init__(self, model_path="weights/yolo26n_traffic_best.pt", conf_thresh=0.5, nms_thresh=0.7):
        """
        Khởi tạo YOLODetector hỗ trợ đầy đủ 6 lớp đối tượng giao thông.

        Args:
            model_path (str): Đường dẫn tới file trọng số .pt hoặc mô hình engine.
            conf_thresh (float): Ngưỡng tự tin (confidence threshold) lọc Bounding Box.
            nms_thresh (float): Ngưỡng Non-Maximum Suppression (NMS IoU). Mặc định 0.7 (70%).
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.img_width = 640
        self.img_height = 384

        # Tự động chuyển fallback nếu model_path chưa tồn tại
        if not os.path.exists(model_path):
            candidates = [
                "weights/yolo26n_traffic_best.pt",
                "weights/yolo26n_best.pt",
                "weights/best.pt",
                "weights/yolo26n.pt",
            ]
            for c in candidates:
                if os.path.exists(c):
                    print(f"[INFO] '{model_path}' không tồn tại, tự động chuyển sang: {c}")
                    model_path = c
                    break

        self.model_path = model_path
        self.is_pt = model_path.endswith(".pt") or model_path.endswith(".engine")

        if self.is_pt:
            from ultralytics import YOLO
            print(f"[INFO] Nạp mô hình từ: {model_path}")
            self.model = YOLO(model_path)
            self.num_classes = len(self.model.names) if hasattr(self.model, "names") else 6
            print(f"[SUCCESS] Đã nạp YOLO ({self.num_classes} classes trong checkpoint)!")
            if self.num_classes == 6:
                print(f"[INFO] Đang sử dụng mô hình Fine-tuned 6 lớp giao thông: {list(self.TRAFFIC_CLASSES.values())}")
            else:
                print(f"[INFO] Đang sử dụng mô hình gốc COCO, tự động lọc và ánh xạ sang 6 lớp giao thông.")
            return

        raise NotImplementedError(f"Định dạng mô hình không được hỗ trợ hoặc không tồn tại: {model_path}")

    def detect(self, image):
        """
        Thực hiện toàn bộ pipeline nhận diện: Tiền xử lý -> Suy luận -> Hậu xử lý.

        Args:
            image (numpy.ndarray): Khung hình thô gốc (BGR).

        Returns:
            tuple: Trả về 3 mảng numpy:
                - bboxes (numpy.ndarray): Tọa độ hộp bao [x_top_left, y_top_left, width, height], shape (N, 4).
                - confs (numpy.ndarray): Điểm tự tin của các nhận diện, shape (N,).
                - class_ids (numpy.ndarray): ID phân loại đối tượng (0-5), shape (N,).
                Nếu không phát hiện được gì, trả về các mảng rỗng.
        """
        if self.num_classes == 6:
            query_classes = [0, 1, 2, 3, 4, 5]
            map_cls = lambda c: c
        else:
            query_classes = list(self.COCO_TO_TRAFFIC.keys())  # [0, 1, 2, 3, 5, 7]
            map_cls = lambda c: self.COCO_TO_TRAFFIC.get(c, -1)

        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.nms_thresh,
            classes=query_classes,
            imgsz=[self.img_height, self.img_width],
            verbose=False,
        )

        final_bboxes = []
        final_confs = []
        final_class_ids = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                raw_cls_id = int(box.cls[0].cpu().numpy())
                target_cls_id = map_cls(raw_cls_id)

                if 0 <= target_cls_id <= 5:
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    if w > 0 and h > 0:
                        final_bboxes.append([float(x1), float(y1), w, h])
                        final_confs.append(conf)
                        final_class_ids.append(int(target_cls_id))

        if not final_bboxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.array(final_bboxes, dtype=np.float32),
            np.array(final_confs, dtype=np.float32),
            np.array(final_class_ids, dtype=np.int32),
        )
