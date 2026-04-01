# Class trích xuất đặc trưng ngoại hình.
import torch
import cv2
import numpy as np
import torchvision.transforms as T
import os 

from models.osnet import osnet_x0_25

class ReIDExtractor:
    """Trích xuất đặc trưng ngoại hình từ ảnh người dùng.
    Sử dụng mô hình OSNet.
    Input: Ảnh cắt của phương tiện(người hoặc xe) đã được chuẩn hóa.
    Output: Vector đặc trưng ngoại hình 512 chiều.
    """
    def __init__(self, model_path="weights/osnet_x0_25.pth", device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if device is None else device
        print(f"[INFO] Khởi tạo OSNet Extractor: {self.device.upper()}")

        # Khởi tạo mô hình OSNet x0.25
        self.model = osnet_x0_25()

        # Tải weights nếu chưa tồn tại
        if not os.path.exists(model_path):
            print(f"[!] Không tìm thấy file {model_path}. Đang tải...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://huggingface.co/spaces/rachana219/MODT2/resolve/main/trackers/strongsort/deep/checkpoint/osnet_x0_25_msmt17.pth"
            torch.hub.download_url_to_file(url, model_path)
            print(f"[INFO] Tải xong {model_path}")

        # Load trọng số vào mô hình
        state_dict = torch.load(model_path, map_location=self.device)
        # Chỉ lấy vector đăc trưng
        state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Tiền sử lý ảnh kích thước 256x128 và chuẩn hóa
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_image_patches(self, image, bboxes):
        """Cắt ảnh theo bounding boxes và chuẩn hóa."""
        patches = []
        h_img, w_img = image.shape[:2]

        for box in bboxes:
            x, y, w, h = box[:4]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_img, int(x + w)), min(h_img, int(y + h))
            patch = image[y1:y2, x1:x2]

            if patch.size == 0:
                patch = np.zeros((256, 128, 3), dtype=np.uint8)  # Patch trống nếu bbox không hợp lệ
            patches.append(patch)

        return patches
    
    def extract(self, image, bboxes):
        """Trích xuất đặc trưng ngoại hình từ ảnh và bounding boxes."""
        if len(bboxes) == 0:
            return np.empty((0, 512), dtype=np.float32)  # Trả về mảng rỗng nếu không có bbox
        
        patches = self._extract_image_patches(image, bboxes)

        tensors = []

        for patch in patches:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            tensor = self.transform(patch_rgb)
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model(batch)

        return features.cpu().numpy().astype(np.float32)

# Test code
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.kitti_parser import KittiParser

    test_dir = "datasets/KITTI_MOT/KITTI-0000"
    if os.path.exists(test_dir):
        parser = KittiParser(test_dir)
        extractor = ReIDExtractor()
        
        for frame_idx, img, dets in parser.get_frame():
            bboxes = dets[:, :4]  # Lấy bounding boxes
            print(f"\n[Frame {frame_idx:04d}]")
            features = extractor.extract(img, bboxes)
            print(f"-> Số lượng xe: {len(bboxes)}")
            print(f"-> Kích thước Vector Đặc trưng: {features.shape} (N, 512)")
            print(f"-> L2 Norm (Phải = 1.0): {np.linalg.norm(features[0]):.4f}")
            break