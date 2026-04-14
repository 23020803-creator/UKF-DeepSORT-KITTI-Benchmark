import cv2
import numpy as np
from openvino import Core, AsyncInferQueue

class ReIDExtractor:
    def __init__(self, model_path, device="CPU"):
        self.ie = Core()
        self.model = self.ie.read_model(model=model_path)
        
        input_shape = self.model.input(0).partial_shape
        self.c = input_shape[1].get_length() if input_shape[1].is_static else 3
        self.h = input_shape[2].get_length() if input_shape[2].is_static else 256
        self.w = input_shape[3].get_length() if input_shape[3].is_static else 128 
        
        # 1. TRỞ LẠI STATIC SHAPE (BATCH = 1) -> Khử hoàn toàn độ trễ cấp phát RAM
        self.model.reshape([1, self.c, self.h, self.w])
        
        # 2. CẤU HÌNH ĐA LUỒNG: THROUGHPUT sẽ mở tối đa số luồng (Streams) CPU
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=device, config=config)
        
        # 3. KHỞI TẠO HÀNG ĐỢI BẤT ĐỒNG BỘ (Vũ khí tối thượng)
        self.infer_queue = AsyncInferQueue(self.compiled_model)
        
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        out_shape = self.output_layer.partial_shape
        self.output_dim = out_shape[-1].get_length() if out_shape[-1].is_static else 512 

        self.is_osnet = "osnet" in model_path.lower()

    def _preprocess(self, img):
        resized = cv2.resize(img, (self.w, self.h))
        
        if self.is_osnet:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            resized = (resized - mean) / std
        else:
            resized = resized.astype(np.float32)

        input_data = resized.transpose(2, 0, 1) 
        return input_data.reshape(1, self.c, self.h, self.w) # Đóng gói lại thành [1, C, H, W]

    def extract(self, image, bboxes):
        num_bboxes = len(bboxes) if bboxes is not None else 0
        if num_bboxes == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)

        # Ma trận rỗng chờ chứa kết quả
        features = np.zeros((num_bboxes, self.output_dim), dtype=np.float32)
        valid_indices = []

        # 4. HÀM CALLBACK: Khi 1 luồng CPU tính xong 1 xe, nó tự động gọi hàm này để lưu kết quả
        def callback(request, userdata):
            idx = userdata
            res = request.get_output_tensor(0).data[0]
            features[idx] = res

        self.infer_queue.set_callback(callback)

        # 5. NÉM VIỆC CHO CPU: Duyệt qua các xe, tiền xử lý và ném ngay vào Hàng đợi
        for idx, bbox in enumerate(bboxes):
            x, y, w, h = [int(v) for v in bbox]
            crop = image[max(0, y):min(image.shape[0], y+h), 
                         max(0, x):min(image.shape[1], x+w)]
            if crop.size > 0:
                input_blob = self._preprocess(crop)
                
                # Hàm này ném việc vào queue và lập tức quay lại vòng for (Không block Python)
                self.infer_queue.start_async({0: input_blob}, userdata=idx)
                valid_indices.append(idx)

        # 6. CHỜ KẾT QUẢ: Bắt Python đợi cho đến khi tất cả các luồng CPU làm xong việc
        self.infer_queue.wait_all()

        # 7. L2 Normalization chuẩn
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.divide(features, norms, out=np.zeros_like(features), where=norms > 1e-6)
            
        return features