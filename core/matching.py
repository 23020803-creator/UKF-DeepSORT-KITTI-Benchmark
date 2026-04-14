# Toán học: Cosine Distance, IoU, Thuật toán Hungarian.
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def compute_cosine_distance(tracks, features_new):
    """Tính khoảng cách Cosine từ Det mới tới TOÀN BỘ góc nhìn trong Ngân hàng đặc trưng của Track"""
    if len(tracks) == 0 or len(features_new) == 0:
        return np.empty((len(tracks), len(features_new)))
    
    cost_matrix = np.zeros((len(tracks), len(features_new)))
    for i, track in enumerate(tracks):
        gallery = track.get_features() # Ma trận các góc nhìn đã lưu
        if len(gallery) == 0:
            cost_matrix[i, :] = 1.0
            continue
            
        # So sánh N det mới với M góc nhìn trong gallery
        distances = cdist(gallery, features_new, metric='cosine')
        
        # Lấy khoảng cách NHỎ NHẤT (giống nhất với 1 góc nhìn bất kỳ từng thấy)
        cost_matrix[i, :] = np.min(distances, axis=0)
        
    return cost_matrix

def compute_iou_matrix(tracks, detections, img_w, img_h):
    """Tính ma trận IoU có hỗ trợ Asymmetrical IoM khi xe chạm viền ảnh"""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, trk in enumerate(tracks):
        box1 = trk.to_tlwh() # Box UKF dự đoán
        for j, det in enumerate(detections):
            box2 = det.bbox # Box YOLO thực tế
            
            x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
            x_right, y_bottom = min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                cost_matrix[i, j] = 1.0
                continue

            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
            iou = intersection / float(area1 + area2 - intersection + 1e-6)
            
            # --- ĐÃ SỬA: MỞ KHÓA LUẬT IOM CHO TOÀN MÀN HÌNH ---
            # Bất kể xe bị che bởi lề ảnh hay bởi gốc cây giữa đường, 
            # cứ giãn nở đột ngột là dùng IoM để cứu!
            min_area = min(area1, area2)
            iom = intersection / float(min_area + 1e-6)
            
            # Nếu box nhỏ nằm lọt thỏm > 80% trong box to -> Tha thứ sự giãn nở
            if iom > 0.8:
                best_metric = max(iou, iom)
            else:
                best_metric = iou 
                
            cost_matrix[i, j] = 1.0 - best_metric
            
    return cost_matrix

def linear_assignment(cost_matrix, tracks, detections, max_distance=0.2):
    """
    Giải quyết bài toán ghép cặp (Bipartite Matching) bằng thuật toán Hungarian.
    
    Thuật toán Hungarian (linear_sum_assignment) tìm ra cách ghép sao cho 
    *tổng* chi phí của toàn bộ các cặp là nhỏ nhất (Global Optimum), 
    thay vì ưu tiên ghép cặp có chi phí thấp nhất trước (Greedy).
    
    Args:
        cost_matrix: Ma trận (M, N) từ hàm compute_cosine_distance.
        tracks: Danh sách đối tượng Track (M). Yêu cầu đối tượng có thuộc tính `class_id`.
        detections: Danh sách đối tượng Detection (N). Yêu cầu đối tượng có thuộc tính `class_id`.
        max_distance: Ngưỡng loại bỏ. Khoảng cách > max_distance sẽ không được ghép.
        
    Returns:
        matches: List các tuple (track_idx, det_idx) ghép thành công.
        unmatched_tracks: List các track_idx không ghép được.
        unmatched_detections: List các det_idx mới tinh.
    """
    # Khởi tạo kết quả rỗng nếu ma trận trống
    if cost_matrix.size == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    # =========================================================================
    # BƯỚC 1: COST GATING (LỌC THEO CLASS ID)
    # Không bao giờ cho phép ghép khác loại (VD: Track là Người, Det là Xe máy).
    # Gán chi phí vô cùng lớn để thuật toán Hungarian tự động né các cặp này.
    # =========================================================================
    INFTY_COST = 1e+5
    for r, track in enumerate(tracks):
        for c, det in enumerate(detections):
            if track.class_id != det.class_id:
                cost_matrix[r, c] = INFTY_COST

    # =========================================================================
    # BƯỚC 2: CHẠY THUẬT TOÁN HUNGARIAN
    # Trả về 2 mảng: row_indices (Track) và col_indices (Detection)
    # tương ứng với các cặp có tổng chi phí nhỏ nhất.
    # =========================================================================
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = []
    unmatched_detections = []

    # =========================================================================
    # BƯỚC 3: PHÂN LOẠI KẾT QUẢ VÀ ÁP DỤNG NGƯỠNG (MAX DISTANCE)
    # =========================================================================
    
    # 3.1. Tìm các đối tượng bị thuật toán Hungarian bỏ rơi ngay từ đầu
    # (Do số lượng M và N chênh lệch nhau)
    unmatched_tracks = list(set(range(len(tracks))) - set(row_indices))
    unmatched_detections = list(set(range(len(detections))) - set(col_indices))

    # 3.2. Lọc lại các cặp Hungarian đã ghép xem có thỏa mãn ngưỡng không
    for row, col in zip(row_indices, col_indices):
        # Nếu chi phí > ngưỡng cho phép (hoặc chạm mức INFTY do sai Class ID)
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(row)
            unmatched_detections.append(col)
        else:
            matches.append((row, col))

    return matches, unmatched_tracks, unmatched_detections