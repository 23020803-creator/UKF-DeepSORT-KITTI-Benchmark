"""
MODULE: OpenVINO ReID Extractor (Robust Version)
-----------------------------------------------
Mô tả:
    Lớp OpenVINOReIDExtractor cung cấp giải pháp trích xuất đặc trưng (Embeddings) 
    từ ảnh đối tượng bằng OpenVINO Toolkit. Phiên bản này được thiết kế để tự động 
    thích ứng với cấu trúc Shape của model (Dynamic/Static).

Chức năng chính:
    1.  Khởi tạo (Constructor): Nạp model, tự động truy vấn kích thước đầu vào (N, C, H, W) 
        để thiết lập quy trình tiền xử lý tương ứng.
    2.  Tiền xử lý (Preprocess): Thực hiện Resize, chuẩn hóa định dạng Tensor (HWC -> CHW) 
        và thêm chiều Batch (1, C, H, W).
    3.  Trích xuất (Extract): Cắt vùng ảnh đối tượng, chạy suy luận và thực hiện 
        L2 Normalization để thu được vector đặc trưng chuẩn hóa.

Đầu vào (Inputs):
    - image (np.ndarray): Ảnh gốc (BGR) từ camera hoặc video, shape (H, W, 3).
    - bboxes (np.ndarray/list): Danh sách các tọa độ vùng chứa đối tượng [x, y, w, h].

Đầu ra (Outputs):
    - features (np.ndarray): Mảng 2D chứa các vector đặc trưng, shape (N, feature_dim).
      Mỗi hàng là một "định danh số" của đối tượng, dùng để so sánh (matching).

Đặc điểm kỹ thuật cần lưu ý:
    - Xử lý Dynamic Shape: Sử dụng `partial_shape` để lấy thông tin model ngay cả khi 
      kích thước chưa được xác định cứng (Static) lúc export model.
    - L2 Normalization: Áp dụng công thức v = v / (||v|| + 1e-6) để tránh lỗi chia cho 0 
      và đưa vector về không gian Euclid chuẩn.
"""

import cv2
import numpy as np
from openvino.runtime import Core

class OpenVINOReIDExtractor:
    def __init__(self, model_path, device="CPU"):
        print(f"[INFO] Khởi tạo OpenVINO Extractor trên thiết bị: {device}")
        self.ie = Core()
        
        # 1. Đọc mô hình (.xml hoặc .onnx đều được)
        self.model = self.ie.read_model(model=model_path)
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=device)
        
        # 2. Lấy thông tin đầu vào/đầu ra
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # 3. Xử lý Dynamic Shape (Quan trọng để không bị crash)
        shape = self.input_layer.partial_shape
        self.n = shape[0].get_length() if shape[0].is_static else 1
        self.c = shape[1].get_length() if shape[1].is_static else 3
        self.h = shape[2].get_length() if shape[2].is_static else 256 # Mặc định ReID thường là 256
        self.w = shape[3].get_length() if shape[3].is_static else 128 # hoặc 256 tùy model
        
        print(f"[INFO] Model Loaded! Input shape: {self.n}x{self.c}x{self.h}x{self.w}")

    def preprocess(self, img):
        """Tiền xử lý ảnh đúng chuẩn OpenVINO"""
        # Resize về kích thước model yêu cầu
        resized_img = cv2.resize(img, (self.w, self.h))
        # Đổi từ HWC (ảnh OpenCV) sang CHW (ảnh Tensor)
        input_img = resized_img.transpose(2, 0, 1)
        # Thêm chiều Batch (1, C, H, W)
        input_img = input_img.reshape(1, self.c, self.h, self.w)
        return input_img.astype(np.float32)

    def extract(self, image, bboxes):
        """Trích xuất vector đặc trưng cho một danh sách các Bounding Boxes"""
        features = []
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            # Cắt ảnh đối tượng (Crop)
            crop = image[y:y+h, x:x+w]
            if crop.size == 0:
                features.append(np.zeros(self.output_layer.shape[-1]))
                continue
                
            # Tiền xử lý và chạy Inference
            blob = self.preprocess(crop)
            result = self.compiled_model([blob])[self.output_layer]
            
            # Làm phẳng vector và chuẩn hóa (L2 Normalization)
            feat = result.flatten()
            feat /= np.linalg.norm(feat) + 1e-6
            features.append(feat)
            
        return np.array(features)