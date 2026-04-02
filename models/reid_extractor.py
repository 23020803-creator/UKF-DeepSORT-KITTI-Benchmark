import cv2
import numpy as np
from openvino.runtime import Core

class ReIDExtractor:
    def __init__(self, model_path, device="CPU"):
        self.ie = Core()
        self.model = self.ie.read_model(model=model_path)
        
        # 1. Trích xuất Shape an toàn (Tránh lỗi Crash lúc khởi tạo)
        input_shape = self.model.input(0).partial_shape
        self.c = input_shape[1].get_length() if input_shape[1].is_static else 3
        self.h = input_shape[2].get_length() if input_shape[2].is_static else 256
        self.w = input_shape[3].get_length() if input_shape[3].is_static else 256
        
        # 2. KHÔNG DÙNG PADDING NỮA. Khai báo Batch Size là Động (-1)
        # Báo cho OpenVINO biết số lượng ảnh sẽ thay đổi linh hoạt.
        self.model.reshape([-1, self.c, self.h, self.w])
        
        # 3. Ưu tiên độ trễ thấp nhất cho luồng Tracking
        config = {"PERFORMANCE_HINT": "LATENCY"}
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=device, config=config)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        out_shape = self.output_layer.partial_shape
        self.output_dim = out_shape[-1].get_length() if out_shape[-1].is_static else 512 

    def _preprocess(self, img):
        resized = cv2.resize(img, (self.w, self.h))
        input_data = resized.transpose(2, 0, 1) 
        input_data = input_data.reshape(1, self.c, self.h, self.w).astype(np.float32)
        return input_data

    def extract(self, image, bboxes):
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)

        blobs = []
        valid_indices = []

        # Cắt và tiền xử lý từng xe
        for idx, bbox in enumerate(bboxes):
            x, y, w, h = [int(v) for v in bbox]
            crop = image[max(0, y):min(image.shape[0], y+h), 
                         max(0, x):min(image.shape[1], x+w)]
            if crop.size > 0:
                blob = self._preprocess(crop) 
                blobs.append(blob)
                valid_indices.append(idx)
                
        if len(blobs) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)

        # Gộp Mảng (Batching): Có 3 xe thì mảng là (3, C, H, W). Không đệm thêm rác!
        batch_blob = np.concatenate(blobs, axis=0)
        
        # CPU chỉ tính toán đúng số lượng xe có thật trên màn hình
        results = self.compiled_model([batch_blob])[self.output_layer] 
        
        # Chuẩn hóa L2 tốc độ cao bằng Vectorization
        features = np.zeros((len(bboxes), self.output_dim), dtype=np.float32)
        norms = np.linalg.norm(results, axis=1, keepdims=True)
        normalized_results = np.divide(results, norms, out=np.zeros_like(results), where=norms > 1e-6)
        
        for i, v_idx in enumerate(valid_indices):
            features[v_idx] = normalized_results[i]
            
        return features