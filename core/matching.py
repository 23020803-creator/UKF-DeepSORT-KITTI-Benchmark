# Toán học: Cosine Distance, IoU, Thuật toán Hungarian.
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.linalg

def compute_cosine_distance(tracks, features_new):
    """Tính khoảng cách Cosine từ Det mới tới Track (Đã tối ưu cho EMA)"""
    if len(tracks) == 0 or len(features_new) == 0:
        return np.empty((len(tracks), len(features_new)))
    
    # 1. Rút trích tất cả smooth_feature của các Track hiện tại thành ma trận (M, 512)
    track_features = []
    for track in tracks:
        # Lấy feature, nếu bị None (do khởi tạo chưa có) thì dùng vector 0
        feat = track.smooth_feature if (hasattr(track, 'smooth_feature') and track.smooth_feature is not None) else np.zeros(512)
        track_features.append(feat)
        
    track_features = np.array(track_features)
    
    # 2. Tính toán Vectorized cực nhanh cho toàn bộ hệ thống
    # cdist(M x 512, N x 512) -> Trả ra ma trận (M, N)
    cost_matrix = cdist(track_features, features_new, metric='cosine')
    
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

def compute_mahalanobis_distance(tracks, detections):
    """
    Tính ma trận bình phương khoảng cách Mahalanobis (Dynamic Gating).
    Trả về ma trận (M, N).
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    dist_matrix = np.zeros((len(tracks), len(detections)))
    
    for i, trk in enumerate(tracks):
        # === BẮT ĐẦU PHẦN SỬA ĐỔI ===
        # 1. Trích xuất State thô
        raw_state = trk.ukf.mean
        cx, cy, a, h = raw_state[0], raw_state[1], raw_state[2], raw_state[3]
        w = a * h
        
        # 2. Chiếu State thô sang Measurement Space (Giả lập lại h(x))
        xmin = max(0.0, cx - w / 2.0)
        xmax = min(1242.0, cx + w / 2.0)
        ymin = max(0.0, cy - h / 2.0)
        ymax = min(375.0, cy + h / 2.0)
        
        # Nếu xe văng hẳn ra ngoài, đánh chi phí cực đại để tránh chia 0
        if xmin >= xmax or ymin >= ymax:
            dist_matrix[i, :] = 1e5
            continue
            
        w_obs = xmax - xmin
        h_obs = ymax - ymin
        cx_obs = (xmin + xmax) / 2.0
        cy_obs = (ymin + ymax) / 2.0
        a_obs = w_obs / h_obs if h_obs > 0 else a
        
        # Đây mới là Không gian Đo lường thực sự mà YOLO nhìn thấy!
        mean_proj = np.array([cx_obs, cy_obs, a_obs, h_obs])
        # === KẾT THÚC PHẦN SỬA ĐỔI ===
        
        P_proj = trk.ukf.covariance[:4, :4]
        
        # 2. Tính ma trận nhiễu đo lường R (Đồng bộ với ukf.py)
        h = max(trk.ukf.mean[3], 1.0)
        std_pos = 0.02 * h
        R = np.zeros((4, 4), dtype=np.float64)
        R[0, 0] = R[1, 1] = R[3, 3] = std_pos ** 2
        R[2, 2] = 1e-4
        
        # 3. Tính Innovation Covariance S = P + R
        S = P_proj + R
        try:
            # Dùng Cholesky decomposition để giải nghịch đảo ma trận an toàn & nhanh hơn
            cholesky_factor = scipy.linalg.cho_factor(S)
        except np.linalg.LinAlgError:
            # Fallback an toàn nếu hệ không xác định dương (Positive-definite)
            S = S + np.eye(4) * 1e-6
            cholesky_factor = scipy.linalg.cho_factor(S)
            
        for j, det in enumerate(detections):
            # Biến đổi bbox YOLO về [cx, cy, a, h]
            z = trk._tlwh_to_xyah(det.bbox)
            innovation = z - mean_proj
            
            # Tính D^2 = innovation^T * S^-1 * innovation
            y = scipy.linalg.cho_solve(cholesky_factor, innovation)
            sq_maha_dist = np.dot(innovation, y)
            
            dist_matrix[i, j] = sq_maha_dist
            
    return dist_matrix