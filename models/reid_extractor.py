"""
MODULE: ReID Feature Extractor (OpenVINO Implementation)
-------------------------------------------------------
Mô tả:
    Lớp ReIDExtractor thực hiện nhiệm vụ trích xuất đặc trưng (Feature Extraction) 
    từ các vùng ảnh đối tượng (đã được detection trước đó). Đây là bước then chốt 
    trong các bài toán Person Re-Identification hoặc Object Tracking (như DeepSORT).

Nguyên lý hoạt động:
    1. Cắt (Crop) các vùng ảnh từ ảnh gốc dựa trên Bounding Boxes.
    2. Tiền xử lý (Preprocessing): Resize về kích thước đầu vào của model và 
       chuyển đổi format từ HWC (OpenCV) sang CHW (OpenVINO/Tensor).
    3. Suy luận (Inference): Sử dụng OpenVINO để chạy mô hình Deep Learning.
    4. Hậu xử lý (Post-processing): Làm phẳng (flatten) và chuẩn hóa L2 (L2 Normalization).

Đầu vào (Inputs):
    - image (np.ndarray): Ảnh gốc định dạng BGR, shape (H, W, 3).
    - bboxes (np.ndarray): Mảng các khung bao định dạng [x1, y1, w, h], shape (N, 4).

Đầu ra (Outputs):
    - features (np.ndarray): Mảng các vector đặc trưng đã được chuẩn hóa L2, 
      shape (N, vector_dim). Thông thường vector_dim = 512 hoặc 256 tùy model.

Tại sao phải chuẩn hóa L2?
    Việc chuẩn hóa v = v / ||v|| đưa các vector về cùng một độ dài (đường tròn đơn vị). 
    Điều này giúp việc so sánh độ tương đồng giữa hai đối tượng trở nên đơn giản hơn:
    Khoảng cách Euclidean lúc này tỉ lệ thuận với khoảng cách Cosine.
"""

import cv2
import numpy as np
from openvino.runtime import Core

class ReIDExtractor:
    def __init__(self, model_path, device="CPU"):
        """
        Khởi tạo bộ trích xuất đặc trưng sử dụng OpenVINO.
        """
        self.ie = Core()
        # Đọc và biên dịch mô hình (.xml)
        self.model = self.ie.read_model(model=model_path)
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=device)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Lấy kích thước chuẩn từ mô hình (thường là 256x128 cho người)
        _, self.c, self.h, self.w = self.input_layer.shape
        self.output_dim = self.output_layer.shape[-1] 
        
        print(f"[INFO] Đã nạp ReID model thành công. Output Dimension: {self.output_dim}")

    def _preprocess(self, img):
        """Tiền xử lý ảnh đúng chuẩn Tensor CHW"""
        resized = cv2.resize(img, (self.w, self.h))
        input_data = resized.transpose(2, 0, 1) # HWC -> CHW
        input_data = input_data.reshape(1, self.c, self.h, self.w).astype(np.float32)
        return input_data

    def extract(self, image, bboxes):
        """
        Input: 
            image: Ảnh gốc (H, W, 3)
            bboxes: Numpy array (N, 4) - dạng [x1, y1, w, h]
        Output:
            features: Numpy array shape (N, 512), dtype float32, L2 Normalized.
        """
        # Trường hợp N = 0: Trả về mảng rỗng đúng kích thước (0, 512)
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)

        features = []
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            
            # Cắt ảnh đối tượng (Crop) và đảm bảo không văng khỏi biên ảnh
            crop = image[max(0, y):min(image.shape[0], y+h), 
                         max(0, x):min(image.shape[1], x+w)]
            
            if crop.size == 0:
                # Nếu crop lỗi, trả về vector 0
                features.append(np.zeros(self.output_dim, dtype=np.float32))
                continue
                
            # Chạy Inference
            blob = self._preprocess(crop)
            result = self.compiled_model([blob])[self.output_layer]
            
            # Trích xuất và làm phẳng vector
            feat = result.flatten().astype(np.float32)
            
            # CHUẨN HÓA L2 (L2 Normalization)
            # Công thức: v = v / sqrt(sum(v_i^2))
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            
            features.append(feat)
            
        return np.array(features, dtype=np.float32)