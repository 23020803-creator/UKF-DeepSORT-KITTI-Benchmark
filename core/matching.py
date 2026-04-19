import numpy as np
import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment
import scipy.linalg

def compute_iou_matrix(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        trk_bbox = trk.ukf.x[:4]
        trk_tlwh = _xyah_to_tlwh(trk_bbox)
        for j, det in enumerate(detections):
            det_tlwh = det.tlwh
            iou = _calculate_iou(trk_tlwh, det_tlwh)
            cost_matrix[i, j] = 1.0 - iou
    return cost_matrix

def compute_cosine_distance(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    
    trk_features = np.array([trk.smooth_feature for trk in tracks], dtype=np.float32)
    det_features = np.array([det.feature for det in detections], dtype=np.float32)
    
    # Vectorized cosine distance computation
    cost_matrix = scipy.spatial.distance.cdist(trk_features, det_features, metric='cosine')
    return cost_matrix

def compute_mahalanobis_distance(tracks, detections):
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, trk in enumerate(tracks):
        mean_proj = trk.ukf.x[:4]
        P_proj = trk.ukf.P[:4, :4]
        
        # mean_proj is [cx, cy, a, h], index 3 is height
        h = max(mean_proj[3], 1.0) 
        
        std_pos = 0.05 * h
        # NSA Noise Scaling is inherently handled inside track UKF update, 
        # for Mahalanobis prediction we use the base measurement noise scale
        R = np.diag([std_pos**2, std_pos**2, 1e-2, std_pos**2])
        S = P_proj + R
        
        try:
            cho_factor, lower = scipy.linalg.cho_factor(S, check_finite=False)
        except scipy.linalg.LinAlgError:
            S += np.eye(4) * 1e-6
            cho_factor, lower = scipy.linalg.cho_factor(S, check_finite=False)
            
        for j, det in enumerate(detections):
            z = det.xyah
            innovation = z - mean_proj
            dist = scipy.linalg.cho_solve((cho_factor, lower), innovation, check_finite=False)
            cost_matrix[i, j] = np.dot(innovation.T, dist)
    
    cost_matrix[np.isinf(cost_matrix)] = 1e5
    return cost_matrix

def fuse_score(cost_matrix, tracks, detections):
    """ Dung hợp Cost Matrix theo kiểu BoT-SORT """
    if cost_matrix.size == 0:
        return cost_matrix
    
    iou_dist = compute_iou_matrix(tracks, detections)
    # Gating ReID: Từ chối các kết nối có khoảng cách Cosine quá lớn
    cost_matrix[cost_matrix > 0.35] = np.inf
    # Sử dụng Minimum Fusion để giữ liên kết nếu một trong hai chỉ số (IoU hoặc Cosine) xuất sắc
    fused_cost = np.minimum(cost_matrix, iou_dist)
    return fused_cost

def linear_assignment(cost_matrix, tracks, detections, max_distance=0.8):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(tracks)), np.arange(len(detections))
    
    # Class ID Gating: Ép buộc cost = vô cực nếu khác lớp đối tượng
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            if trk.class_id != det.class_id:
                cost_matrix[i, j] = np.inf
                
    # === SỬA LỖI INFEASIBLE TRIỆT ĐỂ Ở ĐÂY ===
    # Quét sạch toàn bộ NaN, Inf, -Inf do lỗi chia cho 0 (từ ReID hoặc IoU)
    cost_matrix = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=1e5)
    # =========================================
                
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches, unmatched_tracks, unmatched_detections = [], [], []
    
    for i in range(len(tracks)):
        if i not in row_ind:
            unmatched_tracks.append(i)
    for j in range(len(detections)):
        if j not in col_ind:
            unmatched_detections.append(j)
            
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_distance:
            unmatched_tracks.append(r)
            unmatched_detections.append(c)
        else:
            matches.append((r, c))
            
    return np.array(matches), np.array(unmatched_tracks), np.array(unmatched_detections)

def _xyah_to_tlwh(xyah):
    """ Chuyển đổi từ [center_x, center_y, aspect_ratio, height] sang [top_left_x, top_left_y, width, height] """
    ret = np.asarray(xyah).copy()
    ret[2] *= ret[3]          # width = aspect_ratio * height
    ret[0] -= ret[2] / 2.0    # top_left_x = center_x - width/2
    ret[1] -= ret[3] / 2.0    # top_left_y = center_y - height/2
    return ret

def _calculate_iou(box1, box2):
    """ Tính IoU của 2 bounding box format [top_left_x, top_left_y, width, height] """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou