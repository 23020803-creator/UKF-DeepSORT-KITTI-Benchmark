"""
Mô đun YOLO Detector tối ưu hóa bằng OpenVINO Toolkit.

Cung cấp lõi thực thi suy luận (Inference Engine) cho mạng YOLO11n. 
Tích hợp quy trình tiền xử lý ảnh (Letterbox Resizing) để giữ nguyên tỷ lệ, 
chạy suy luận trực tiếp trên CPU và hậu xử lý NMS (Non-Maximum Suppression).
Chỉ tập trung nhận diện Người (Person) và Xe ô tô (Car).
"""

import os
import cv2
import numpy as np
import torch
import openvino as ov

try:
    from ultralytics.utils.ops import non_max_suppression, scale_boxes
except ImportError:
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes

class YOLODetector:
    """
    Lớp đóng gói mô hình YOLO chạy trên nền tảng OpenVINO.

    Attributes:
        conf_thresh (float): Ngưỡng tự tin tối thiểu để giữ lại Bounding Box.
        img_width (int): Chiều rộng ảnh chuẩn hóa đầu vào của mô hình.
        img_height (int): Chiều cao ảnh chuẩn hóa đầu vào của mô hình.
        core (openvino.runtime.Core): Lõi thực thi OpenVINO.
        compiled_model (openvino.runtime.CompiledModel): Mô hình đã được biên dịch cho thiết bị hiện tại.
        output_layer (openvino.runtime.ConstOutput): Lớp đầu ra của mô hình.
        coco_to_id (dict): Từ điển ánh xạ từ ID của bộ dữ liệu COCO sang ID tùy chỉnh của hệ thống.
    """

    def __init__(self, model_path="weights/yolo11n_int8_openvino_model", conf_thresh=0.7):
        """
        Khởi tạo YOLODetector, nạp và biên dịch mô hình lên CPU.

        Args:
            model_path (str): Đường dẫn tới thư mục chứa tệp mô hình OpenVINO (yolo11n.xml).
            conf_thresh (float): Ngưỡng tự tin (confidence threshold) cho NMS.
        """
        self.conf_thresh = conf_thresh
        
        self.img_width = 960  
        self.img_height = 288 
        
        print("[INFO] Đang khởi tạo Lõi OpenVINO (Native Core)...")
        self.core = ov.Core()
        
        xml_path = os.path.join(model_path, "yolo11n.xml")
        model = self.core.read_model(model=xml_path)
        
        self.compiled_model = self.core.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)
        print("[SUCCESS] Đã load và biên dịch model trực tiếp lên CPU!")

        # Chỉ giữ lại Person (0) và Car (2), loại bỏ các class không cần thiết
        self.coco_to_id = {
            0: 0,  # person -> Person
            2: 2,  # car -> Car
        }

    def _letterbox(self, img, new_shape=(960, 288), color=(114, 114, 114)):
        """
        Thay đổi kích thước ảnh (Resize) và thêm viền (Padding) để giữ nguyên tỷ lệ khung hình.
        
        Quá trình này ngăn chặn sự biến dạng vật thể (squishing/stretching),
        giúp mô hình YOLO nhận diện chính xác hơn.

        Args:
            img (numpy.ndarray): Ảnh BGR đầu vào.
            new_shape (tuple): Kích thước mục tiêu (chiều rộng, chiều cao).
            color (tuple): Mã màu BGR của viền padding. Mặc định là xám (114, 114, 114).

        Returns:
            numpy.ndarray: Ảnh sau khi resize và thêm viền.
        """
        shape = img.shape[:2]  
        
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) 
        
        dw = new_shape[0] - new_unpad[0]  
        dh = new_shape[1] - new_unpad[1]  
        
        dw /= 2  
        dh /= 2  
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img

    def detect(self, image):
        """
        Thực hiện toàn bộ pipeline nhận diện: Tiền xử lý -> Suy luận -> Hậu xử lý.

        Args:
            image (numpy.ndarray): Khung hình thô gốc (BGR).

        Returns:
            tuple: Trả về 3 mảng numpy:
                - bboxes (numpy.ndarray): Tọa độ hộp bao [x_top_left, y_top_left, width, height], shape (N, 4).
                - confs (numpy.ndarray): Điểm tự tin của các nhận diện, shape (N,).
                - class_ids (numpy.ndarray): ID phân loại đối tượng, shape (N,).
                Nếu không phát hiện được gì, trả về các mảng rỗng.
        """
        # 1. TIỀN XỬ LÝ
        img_padded = self._letterbox(image, new_shape=(self.img_width, self.img_height))
        
        img_blob = img_padded[:, :, ::-1].transpose(2, 0, 1)  
        img_blob = np.ascontiguousarray(img_blob)
        img_blob = img_blob.astype(np.float32) / 255.0
        img_blob = np.expand_dims(img_blob, axis=0)

        # 2. INFERENCE (Chạy model qua OpenVINO)
        raw_results = self.compiled_model([img_blob])[self.output_layer]

        # 3. HẬU XỬ LÝ
        preds = torch.from_numpy(raw_results)
        
        # Áp dụng Non-Maximum Suppression (NMS) với ngưỡng IoU là 0.30
        preds = non_max_suppression(preds, conf_thres=self.conf_thresh, iou_thres=0.30, max_det=300)
        
        if len(preds) == 0 or len(preds[0]) == 0:
            return (np.empty((0, 4), dtype=np.float32), 
                    np.empty((0,), dtype=np.float32), 
                    np.empty((0,), dtype=np.int32))

        # Khôi phục tọa độ hộp bao về kích thước ảnh gốc
        preds[0][:, :4] = scale_boxes(img_blob.shape[2:], preds[0][:, :4], image.shape).round()
        
        final_bboxes = []
        final_confs = []
        final_class_ids = []

        for det in preds[0]:
            x1, y1, x2, y2, conf, cls = det[:6]
            coco_cls_id = int(cls.item())
            
            # Chỉ xử lý nếu đối tượng nằm trong danh sách được cấp phép (Person, Car)
            if coco_cls_id not in self.coco_to_id:
                continue
                
            target_id = self.coco_to_id[coco_cls_id]
            
            # Đổi tọa độ từ dạng [x1, y1, x2, y2] sang [x, y, w, h]
            final_bboxes.append([float(x1.item()), float(y1.item()), float(x2.item() - x1.item()), float(y2.item() - y1.item())])
            final_confs.append(float(conf.item()))
            final_class_ids.append(int(target_id))

        if not final_bboxes:
            return (np.empty((0, 4), dtype=np.float32), 
                    np.empty((0,), dtype=np.float32), 
                    np.empty((0,), dtype=np.int32))

        return (np.array(final_bboxes, dtype=np.float32), 
                np.array(final_confs, dtype=np.float32), 
                np.array(final_class_ids, dtype=np.int32))