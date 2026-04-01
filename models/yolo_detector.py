"""
=========================================================================================
FILE: yolo_detector.py
CHỨC NĂNG: Lõi thực thi suy luận (Inference Engine) sử dụng OpenVINO API.
           Tích hợp logic tiền xử lý, hậu xử lý và gộp thực thể (Entity Merging).
-----------------------------------------------------------------------------------------
1. ĐẦU VÀO (INPUT):
    - Ảnh thô (Original Frame): Mảng Numpy (BGR) từ OpenCV.
    - Model Path: Đường dẫn tới thư mục chứa file .xml và .bin của OpenVINO.
    - imgsz: Kích thước mạng yêu cầu (mặc định 480x480).

2. ĐẦU RA (OUTPUT) - Trả về Tuple gồm 3 mảng Numpy:
    - bboxes: Mảng (N, 4) chứa tọa độ [x_min, y_min, width, height] kiểu float32.
    - confs: Mảng (N,) chứa điểm tin cậy (0.0 - 1.0) kiểu float32.
    - class_ids: Mảng (N,) chứa ID lớp đã mapping (0: Pedestrian, 1: Cyclist, 2: Car).

3. CƠ CHẾ XỬ LÝ ĐẶC BIỆT (CORE LOGIC):
    - Letterbox Resizing: Thay đổi kích thước ảnh nhưng giữ nguyên tỷ lệ (aspect ratio) 
      bằng cách thêm padding, giúp tránh làm biến dạng vật thể khi inference.
    - IOA Merging (Intersection over Area): Logic đặc thù để kiểm tra sự đè lấp giữa 
      'Person' và 'Vehicle'. Nếu một người nằm trong vùng của xe đạp/xe máy (IOA > 0.3), 
      hệ thống sẽ gộp lại thành một thực thể duy nhất là 'Cyclist'.
    - COCO Mapping: Chuyển đổi từ 80 lớp mặc định của COCO sang 3 lớp mục tiêu của dự án.

4. HIỆU NĂNG:
    - Sử dụng OpenVINO Native API để tối ưu hóa trên kiến trúc CPU Intel.
    - Tích hợp NMS (Non-Maximum Suppression) để loại bỏ các box trùng lặp.
=========================================================================================
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
    def __init__(self, model_path="weights/yolo11n_int8_openvino_model", conf_thresh=0.7):
        self.conf_thresh = conf_thresh
        self.imgsz = 480
        
        print("Đang khởi tạo Lõi OpenVINO (Native Core)...")
        self.core = ov.Core()
        
        xml_path = os.path.join(model_path, "yolo11n.xml")
        model = self.core.read_model(model=xml_path)
        
        self.compiled_model = self.core.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)
        print("✅ Đã load và biên dịch model trực tiếp lên CPU!")

        # MAPPING MỚI: 0: Person, 1: Cyclist, 2: Car
        # COCO IDs: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.coco_to_id = {
            0: 0,  # person -> Person
            1: 1,  # bicycle -> Cyclist
            2: 2,  # car -> Car
            3: 1,  # motorcycle -> Cyclist
            5: 2,  # bus -> Car
            7: 2   # truck -> Car
        }

    def _letterbox(self, img, new_shape=(480, 480), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def _compute_ioa(self, box_person, box_vehicle):
        x_left = max(box_person[0], box_vehicle[0])
        y_top = max(box_person[1], box_vehicle[1])
        x_right = min(box_person[2], box_vehicle[2])
        y_bottom = min(box_person[3], box_vehicle[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        person_area = (box_person[2] - box_person[0]) * (box_person[3] - box_person[1])
        return intersection_area / person_area if person_area > 0 else 0.0

    def detect(self, image, imgsz=480):
        # 1. TIỀN XỬ LÝ
        img_padded = self._letterbox(image, new_shape=(imgsz, imgsz))
        img_blob = img_padded[:, :, ::-1].transpose(2, 0, 1)  
        img_blob = np.ascontiguousarray(img_blob)
        img_blob = img_blob.astype(np.float32) / 255.0
        img_blob = np.expand_dims(img_blob, axis=0)

        # 2. INFERENCE
        raw_results = self.compiled_model([img_blob])[self.output_layer]

        # 3. HẬU XỬ LÝ
        preds = torch.from_numpy(raw_results)
        preds = non_max_suppression(preds, conf_thres=self.conf_thresh, iou_thres=0.45, max_det=300)
        
        if not len(preds[0]):
            return (np.empty((0, 4), dtype=np.float32), 
                    np.empty((0,), dtype=np.float32), 
                    np.empty((0,), dtype=np.int32))

        preds[0][:, :4] = scale_boxes(img_blob.shape[2:], preds[0][:, :4], image.shape).round()
        
        temp_pedestrians = []
        temp_vehicles = []
        
        for *xyxy, conf, cls in preds[0]:
            coco_cls_id = int(cls.item())
            if coco_cls_id not in self.coco_to_id:
                continue
                
            target_id = self.coco_to_id[coco_cls_id]
            x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
            
            # Logic phân loại tạm thời dựa trên ID mới
            if target_id == 0: # Person
                temp_pedestrians.append([x1, y1, x2, y2, conf.item(), target_id])
            else:              # Cyclist (1) hoặc Car (2)
                temp_vehicles.append([x1, y1, x2, y2, conf.item(), target_id])

        # Quét để gộp Người vào Xe đạp/Xe máy (Entity Merging)
        merged_pedestrian_indices = set()
        for v_idx, vehicle in enumerate(temp_vehicles):
            if vehicle[5] == 1:  # Nếu là Cyclist
                for p_idx, person in enumerate(temp_pedestrians):
                    if p_idx in merged_pedestrian_indices:
                        continue
                    if self._compute_ioa(person[:4], vehicle[:4]) > 0.3:
                        # Cập nhật box xe bao trùm cả người
                        temp_vehicles[v_idx][0] = min(vehicle[0], person[0])
                        temp_vehicles[v_idx][1] = min(vehicle[1], person[1])
                        temp_vehicles[v_idx][2] = max(vehicle[2], person[2])
                        temp_vehicles[v_idx][3] = max(vehicle[3], person[3])
                        temp_vehicles[v_idx][4] = max(vehicle[4], person[4])
                        merged_pedestrian_indices.add(p_idx)

        # TÁCH RA 3 MẢNG RIÊNG BIỆT
        final_bboxes = []
        final_confs = []
        final_class_ids = []

        # Thêm người (không bị gộp)
        for p_idx, person in enumerate(temp_pedestrians):
            if p_idx not in merged_pedestrian_indices:
                x1, y1, x2, y2, conf, cls_id = person
                final_bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
                final_confs.append(float(conf))
                final_class_ids.append(int(cls_id))
                
        # Thêm xe (đã gộp hoặc car)
        for vehicle in temp_vehicles:
            x1, y1, x2, y2, conf, cls_id = vehicle
            final_bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            final_confs.append(float(conf))
            final_class_ids.append(int(cls_id))

        if not final_bboxes:
            return (np.empty((0, 4), dtype=np.float32), 
                    np.empty((0,), dtype=np.float32), 
                    np.empty((0,), dtype=np.int32))

        return (np.array(final_bboxes, dtype=np.float32), 
                np.array(final_confs, dtype=np.float32), 
                np.array(final_class_ids, dtype=np.int32))