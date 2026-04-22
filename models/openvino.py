"""Trích xuất vector đặc trưng (ReID Embeddings) sử dụng OpenVINO Toolkit.

Mô đun này cung cấp lớp `OpenVINOReIDExtractor` được thiết kế để tự động 
thích ứng với cấu trúc Shape của model (Dynamic/Static). Nó thực hiện việc
cắt ảnh đối tượng dựa trên bounding box, tiền xử lý tensor, và chạy suy luận 
để trả về vector đặc trưng đã được chuẩn hóa L2, phục vụ cho bài toán liên 
kết dữ liệu (Data Association) trong Multi-Object Tracking.
"""

import cv2
import numpy as np
from openvino.runtime import Core

class OpenVINOReIDExtractor:
    """
    Lớp trích xuất đặc trưng ngoại hình (ReID) tối ưu hóa bằng OpenVINO.

    Tự động truy vấn kích thước đầu vào của mô hình để thiết lập quy trình 
    tiền xử lý động. Hỗ trợ chạy suy luận trên nhiều thiết bị phần cứng 
    khác nhau (CPU, GPU, VPU) thông qua OpenVINO Inference Engine.

    Attributes:
        ie (openvino.runtime.Core): Đối tượng cốt lõi của OpenVINO.
        model (openvino.runtime.Model): Mô hình định dạng ngầm định (IR).
        compiled_model (openvino.runtime.CompiledModel): Mô hình đã được biên dịch 
            và nạp lên thiết bị phần cứng.
        input_layer (openvino.runtime.ConstOutput): Lớp đầu vào của mô hình.
        output_layer (openvino.runtime.ConstOutput): Lớp đầu ra chứa vector đặc trưng.
        n (int): Kích thước Batch Size của mô hình.
        c (int): Số kênh màu (Channels) đầu vào.
        h (int): Chiều cao (Height) ảnh đầu vào mà mô hình yêu cầu.
        w (int): Chiều rộng (Width) ảnh đầu vào mà mô hình yêu cầu.
    """

    def __init__(self, model_path, device="CPU"):
        """
        Khởi tạo Extractor, nạp mô hình và phân tích cấu trúc đầu vào.

        Đọc mô hình từ đường dẫn chỉ định (.xml hoặc .onnx), biên dịch mô hình 
        lên thiết bị tính toán, và phân tích partial_shape để xử lý các mô hình 
        có kích thước tĩnh (static) hoặc động (dynamic).

        Args:
            model_path (str): Đường dẫn tới tệp mô hình OpenVINO (.xml) hoặc ONNX.
            device (str, optional): Tên thiết bị để chạy suy luận (VD: "CPU", "GPU", "AUTO"). 
                                    Mặc định là "CPU".
        """
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
        """
        Tiền xử lý ảnh gốc thành Tensor chuẩn đầu vào của OpenVINO.

        Thực hiện thay đổi kích thước ảnh (resize) về (W, H), chuyển đổi định 
        dạng kênh màu từ HWC sang CHW, và thêm chiều Batch.

        Args:
            img (numpy.ndarray): Ảnh BGR đầu vào đã được cắt (cropped).

        Returns:
            numpy.ndarray: Tensor đầu vào định dạng (1, C, H, W) kiểu float32.
        """
        # Resize về kích thước model yêu cầu
        resized_img = cv2.resize(img, (self.w, self.h))
        # Đổi từ HWC (ảnh OpenCV) sang CHW (ảnh Tensor)
        input_img = resized_img.transpose(2, 0, 1)
        # Thêm chiều Batch (1, C, H, W)
        input_img = input_img.reshape(1, self.c, self.h, self.w)
        return input_img.astype(np.float32)

    def extract(self, image, bboxes):
        """
        Trích xuất vector đặc trưng (Embeddings) cho một danh sách hộp bao.

        Hàm sẽ cắt từng vùng ảnh tương ứng với từng Bounding Box, tiền xử lý, 
        chạy suy luận, và áp dụng chuẩn hóa L2 (L2 Normalization) để đưa vector 
        về không gian Euclid chuẩn.

        Args:
            image (numpy.ndarray): Khung hình gốc (BGR) từ camera, kích thước (H, W, 3).
            bboxes (numpy.ndarray hoặc list): Danh sách các hộp bao đối tượng 
                định dạng [x, y, w, h].

        Returns:
            numpy.ndarray: Mảng 2D chứa các vector đặc trưng kích thước (N, feature_dim), 
            trong đó N là số lượng bboxes hợp lệ. Trả về vector 0 nếu crop ảnh lỗi.
        """
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