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
        
        self.img_width = 960  
        self.img_height = 288 
        
        print("Đang khởi tạo Lõi OpenVINO (Native Core)...")
        self.core = ov.Core()
        
        xml_path = os.path.join(model_path, "yolo11n.xml")
        model = self.core.read_model(model=xml_path)
        
        self.compiled_model = self.core.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)
        print("✅ Đã load và biên dịch model trực tiếp lên CPU!")

        # [SỬA ĐỔI]: Chỉ giữ lại Person (0) và Car (2), đã loại bỏ xe đạp/xe máy
        self.coco_to_id = {
            0: 0,  # person -> Person
            2: 2,  # car -> Car
        }

    def _letterbox(self, img, new_shape=(960, 288), color=(114, 114, 114)):
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
        # 1. TIỀN XỬ LÝ (Không được xóa phần này)
        img_padded = self._letterbox(image, new_shape=(self.img_width, self.img_height))
        
        img_blob = img_padded[:, :, ::-1].transpose(2, 0, 1)  
        img_blob = np.ascontiguousarray(img_blob)
        img_blob = img_blob.astype(np.float32) / 255.0
        img_blob = np.expand_dims(img_blob, axis=0)

        # 2. INFERENCE (Chạy model qua OpenVINO)
        raw_results = self.compiled_model([img_blob])[self.output_layer]

        # 3. HẬU XỬ LÝ
        preds = torch.from_numpy(raw_results)
        
        # Chỉ siết iou_thres xuống 0.30, giữ nguyên mặc định (agnostic=False)
        preds = non_max_suppression(preds, conf_thres=self.conf_thresh, iou_thres=0.30, max_det=300)
        
        if len(preds) == 0 or len(preds[0]) == 0:
            return (np.empty((0, 4), dtype=np.float32), 
                    np.empty((0,), dtype=np.float32), 
                    np.empty((0,), dtype=np.int32))

        preds[0][:, :4] = scale_boxes(img_blob.shape[2:], preds[0][:, :4], image.shape).round()
        
        # [SỬA ĐỔI]: Logic duyệt bounding box rút gọn mới
        final_bboxes = []
        final_confs = []
        final_class_ids = []

        for det in preds[0]:
            x1, y1, x2, y2, conf, cls = det[:6]
            coco_cls_id = int(cls.item())
            
            # Chỉ xử lý nếu đối tượng nằm trong danh sách Person (0) hoặc Car (2)
            if coco_cls_id not in self.coco_to_id:
                continue
                
            target_id = self.coco_to_id[coco_cls_id]
            
            # Đẩy thẳng kết quả vào mảng final
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