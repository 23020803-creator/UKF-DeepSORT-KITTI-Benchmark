# Toán học: Cosine Distance, IoU, Thuật toán Hungarian.
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def compute_cosine_distance(features_old, features_new):
    """
    Tính ma trận chi phí (cost matrix) dựa trên khoảng cách Cosine.
    
    Tại sao lại dùng Cosine thay vì Euclidean (L2)?
    Trong ReID, hướng của vector đại diện cho sự phân bố đặc trưng (màu sắc, cấu trúc). 
    Độ lớn (magnitude) của vector thường bị ảnh hưởng bởi độ sáng/tương phản.
    Cosine Distance chỉ quan tâm đến "góc" giữa 2 vector, nên bền vững hơn trước thay đổi ánh sáng.
    
    Toán học: Cosine_Distance = 1 - Cosine_Similarity.
    - Trùng hướng hoàn toàn: Sim = 1.0 -> Dist = 0.0
    - Vuông góc (Không liên quan): Sim = 0.0 -> Dist = 1.0
    - Ngược hướng hoàn toàn: Sim = -1.0 -> Dist = 2.0
    
    Args:
        features_old: Mảng numpy shape (M, 512) của các Tracks (Mục tiêu đang theo dõi).
        features_new: Mảng numpy shape (N, 512) của các Detections (Phát hiện mới từ YOLO).
        
    Returns:
        Ma trận cost_matrix shape (M, N).
    """
    # Xử lý edge-case: Nếu không có Track cũ hoặc không có Detection mới
    if len(features_old) == 0 or len(features_new) == 0:
        return np.empty((len(features_old), len(features_new)))
    
    # Sử dụng cdist của scipy. Nó được viết bằng C, tối ưu hóa qua BLAS/LAPACK 
    # nên tính toán cực kỳ nhanh so với việc dùng hai vòng lặp for trong Python.
    cost_matrix = cdist(features_old, features_new, metric='cosine')
    
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