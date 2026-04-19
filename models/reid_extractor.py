import numpy as np
import cv2
from openvino.runtime import Core, AsyncInferQueue

class ReIDExtractor:
    def __init__(self, model_path, device='CPU', max_requests=0):
        ie = Core()
        model = ie.read_model(model=model_path)
        self.input_layer = model.input(0)
        
        # SỬA LỖI DYNAMIC SHAPE: Sử dụng partial_shape thay vì shape
        p_shape = self.input_layer.partial_shape
        self.c = p_shape[1].get_length() if p_shape[1].is_static else 3
        self.h = p_shape[2].get_length() if p_shape[2].is_static else 256
        self.w = p_shape[3].get_length() if p_shape[3].is_static else 256
        
        # Chìa khóa thiết lập tốc độ tối đa cho OpenVINO [14, 17]
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        self.compiled_model = ie.compile_model(model=model, device_name=device, config=config)
        
        # Khởi tạo Async Queue. Bỏ qua số 0 để nó tự động Scale theo số lõi vi xử lý
        self.infer_queue = AsyncInferQueue(self.compiled_model, max_requests)

    def _preprocess(self, image):
        img = cv2.resize(image, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def extract(self, image, bboxes):
        if len(bboxes) == 0:
            return np.empty((0, 512), dtype=np.float32)

        # Cấp phát mảng nhớ liên tục (contiguous block) để các callback ghi kết quả vào
        features = np.zeros((len(bboxes), 512), dtype=np.float32)

        # Định nghĩa Callback trả về ngay lập tức
        def callback(request, userdata):
            idx = userdata
            res = request.get_output_tensor(0).data
            dim = min(res.shape[1], 512)
            features[idx, :dim] = res[0, :dim]

        self.infer_queue.set_callback(callback)

        valid_count = 0
        for i, bbox in enumerate(bboxes):
            x, y, w, h = map(int, bbox)
            x1, y1 = max(0, x), max(0, y)
            # Sửa lỗi logic: thêm [0] vào image.shape để lấy chiều cao (height)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            input_blob = self._preprocess(crop)
            
            # Đưa tác vụ vào luồng C++ chạy nền. Hàm không gây nghẽn (non-blocking)
            self.infer_queue.start_async({0: input_blob}, userdata=i)
            valid_count += 1

        if valid_count > 0:
            # Điểm rào cản luồng duy nhất (Barrier). CPU sẽ tính toán đồng loạt toàn bộ Tensor
            self.infer_queue.wait_all()

        # Áp dụng chuẩn hóa L2 trên toán tử Vector (loại bỏ hoàn toàn For loop)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 
        features = features / norms

        return features