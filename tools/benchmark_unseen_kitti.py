"""
Script Đánh giá Độc lập (Cross-Domain / Unseen Benchmark) cho mô hình Unified ReID
trên tập dữ liệu KITTI Tracking (chưa từng xuất hiện trong quá trình huấn luyện).

Đo đạc 3 tiêu chí cốt lõi:
1. ĐỘ CHÍNH XÁC REID (Rank-1, Rank-5, Rank-10, mAP) trên chuỗi Tracklet thực tế của KITTI (Car & Pedestrian).
2. ĐỘ TÁCH BIỆT KHÔNG GIAN (Cosine Separation) giữa Người và Xe để đảm bảo không bị nhầm lẫn class.
3. TỐC ĐỘ SUY LUẬN TRẦN (Pure Latency & Throughput) trên CPU Intel Core i5.
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
from openvino import Core, AsyncInferQueue

# Đảm bảo đường dẫn gốc dự án
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def preprocess_crop(img, target_size=(128, 128)):
    """
    Tiền xử lý chuẩn RGB + ImageNet Normalization tương thích 100% với kiến trúc đã train.
    """
    resized = cv2.resize(img, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std
    return norm.transpose(2, 0, 1).reshape(1, 3, target_size[1], target_size[0])

def parse_kitti_gt(label_dir, image_base_dir, max_sequences=None):
    """
    Thu thập các tracklet Ground Truth của Car và Pedestrian từ KITTI.
    """
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    if max_sequences:
        label_files = label_files[:max_sequences]
        
    print(f"📂 Đang tải dữ liệu nhãn từ {len(label_files)} chuỗi video KITTI...")
    
    # Cấu trúc: {class_name: {global_track_id: [{'frame': f, 'img_path': p, 'bbox': [x1, y1, x2, y2]}]}}
    car_tracks = {}
    ped_tracks = {}
    
    for lfile in label_files:
        seq_name = os.path.splitext(os.path.basename(lfile))[0]
        img_dir = os.path.join(image_base_dir, seq_name)
        if not os.path.exists(img_dir):
            continue
            
        with open(lfile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                frame_idx = int(parts[0])
                track_id = int(parts[1])
                obj_type = parts[2]
                occluded = int(parts[4])
                
                # Bỏ qua đối tượng không theo dõi hoặc bị che khuất nghiêm trọng
                if track_id < 0 or occluded > 2:
                    continue
                    
                x1, y1, x2, y2 = float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])
                w, h = x2 - x1, y2 - y1
                # Bỏ qua box quá bé không đủ chi tiết nhận dạng
                if w < 24 or h < 24:
                    continue
                    
                img_path = os.path.join(img_dir, f"{frame_idx:06d}.png")
                global_id = f"{seq_name}_{track_id}"
                
                entry = {
                    'frame': frame_idx,
                    'img_path': img_path,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'seq': seq_name
                }
                
                if obj_type in ['Car', 'Van']:
                    car_tracks.setdefault(global_id, []).append(entry)
                elif obj_type in ['Pedestrian', 'Person_sitting']:
                    ped_tracks.setdefault(global_id, []).append(entry)
                    
    return car_tracks, ped_tracks

def extract_features_openvino(crops_data, model_path, device="CPU"):
    """
    Trích xuất vector đặc trưng 256D chuẩn L2 bằng OpenVINO AsyncInferQueue.
    """
    ie = Core()
    model = ie.read_model(model_path)
    compiled = ie.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    queue = AsyncInferQueue(compiled)
    
    total = len(crops_data)
    features = np.zeros((total, 256), dtype=np.float32)
    
    def callback(request, userdata):
        idx = userdata
        res = request.get_output_tensor(0).data[0]
        features[idx] = res
        
    queue.set_callback(callback)
    
    # Tải và tiền xử lý ảnh
    cache_img = {}
    
    start_time = time.perf_counter()
    for idx, item in enumerate(crops_data):
        img_path = item['img_path']
        if img_path not in cache_img:
            # Chỉ giữ vài ảnh gần nhất trong RAM
            if len(cache_img) > 10:
                cache_img.clear()
            cache_img[img_path] = cv2.imread(img_path)
            
        img = cache_img[img_path]
        if img is None:
            continue
            
        x1, y1, x2, y2 = item['bbox']
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        if crop.size == 0:
            continue
            
        blob = preprocess_crop(crop)
        queue.start_async({0: blob}, userdata=idx)
        
    queue.wait_all()
    elapsed = time.perf_counter() - start_time
    
    # L2-normalization
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = np.divide(features, norms, out=np.zeros_like(features), where=norms > 1e-6)
    
    return features, elapsed

def compute_rank_and_map(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=10):
    """
    Tính Rank-1, Rank-5, Rank-10 và mAP theo chuẩn học thuật ReID.
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_cams[q_idx]
        order = indices[q_idx]
        
        # Loại bỏ trường hợp cùng track trên cùng 1 frame hoặc không hợp lệ
        remove = (g_pids[order] == q_pid) & (g_cams[order] == q_cam)
        keep = np.invert(remove)
        
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
            
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    if len(all_cmc) == 0:
        return np.zeros(max_rank), 0.0
        
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / len(all_cmc)
    mAP = float(np.mean(all_AP))
    return all_cmc, mAP

def evaluate_reid_on_tracks(track_dict, class_name, model_path, min_track_len=5, max_samples_per_track=8):
    """
    Đánh giá độ chính xác ReID trên tập các chuỗi tracklet.
    - Query: Ảnh tại frame đầu tiên của mỗi tracklet.
    - Gallery: Các ảnh tại các frame sau của tracklet đó + các tracklet khác.
    """
    valid_tracks = {k: v for k, v in track_dict.items() if len(v) >= min_track_len}
    if len(valid_tracks) < 2:
        print(f"⚠️ Không đủ tracklet dài cho lớp {class_name} (chỉ có {len(valid_tracks)} tracks).")
        return None
        
    query_items = []
    gallery_items = []
    
    q_pids, g_pids = [], []
    q_frames, g_frames = [], []
    
    track_to_id = {k: idx for idx, k in enumerate(valid_tracks.keys())}
    
    for t_key, items in valid_tracks.items():
        tid = track_to_id[t_key]
        # Query lấy ảnh đầu tiên
        query_items.append(items[0])
        q_pids.append(tid)
        q_frames.append(items[0]['frame'])
        
        # Gallery lấy các ảnh tiếp theo (lấy mẫu cách quãng để tăng độ đa dạng góc nhìn)
        step = max(1, len(items) // max_samples_per_track)
        for sub_item in items[1::step]:
            gallery_items.append(sub_item)
            g_pids.append(tid)
            g_frames.append(sub_item['frame'])
            
    print(f"\n🔍 [Lớp {class_name}]: {len(valid_tracks)} Tracks | Query: {len(query_items)} ảnh | Gallery: {len(gallery_items)} ảnh")
    
    # Ghép chung để trích xuất đặc trưng 1 lần
    all_items = query_items + gallery_items
    feats, ext_time = extract_features_openvino(all_items, model_path)
    
    q_feats = feats[:len(query_items)]
    g_feats = feats[len(query_items):]
    
    # Ma trận khoảng cách Cosine: 1.0 - dot_product
    distmat = 1.0 - np.dot(q_feats, g_feats.T)
    
    cmc, mAP = compute_rank_and_map(distmat, np.array(q_pids), np.array(g_pids), np.array(q_frames), np.array(g_frames))
    
    return {
        'class': class_name,
        'tracks': len(valid_tracks),
        'query_cnt': len(query_items),
        'gallery_cnt': len(gallery_items),
        'rank1': cmc[0] * 100,
        'rank5': cmc[4] * 100 if len(cmc) > 4 else cmc[-1] * 100,
        'rank10': cmc[9] * 100 if len(cmc) > 9 else cmc[-1] * 100,
        'mAP': mAP * 100,
        'crops': len(all_items),
        'time': ext_time,
        'fps': len(all_items) / ext_time if ext_time > 0 else 0,
        'sample_feats': feats
    }

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Unified ReID trên tập dữ liệu chưa từng thấy (KITTI)")
    parser.add_argument("--model", type=str, default="weights/unified/unified_reid_int8.xml", help="Đường dẫn file mô hình OpenVINO")
    parser.add_argument("--label_dir", type=str, default="datasets/KITTI/training/label_02", help="Thư mục nhãn KITTI")
    parser.add_argument("--image_dir", type=str, default="datasets/KITTI/training/image_02", help="Thư mục ảnh KITTI")
    parser.add_argument("--max_seq", type=int, default=10, help="Số lượng sequence dùng để đánh giá (mặc định: 10 sequences)")
    args = parser.parse_args()
    
    print("="*75)
    print("🚀 BENCHMARK UNSEEN DATASET: KIỂM CHỨNG UNIFIED ReID TRÊN KITTI TRACKING")
    print("="*75)
    print(f"📦 Mô hình đánh giá : {args.model}")
    print(f"🏷️ Thư mục Ground Truth: {args.label_dir}")
    print(f"🖼️ Thư mục Ảnh KITTI    : {args.image_dir}")
    
    if not os.path.exists(args.model):
        print(f"❌ Không tìm thấy file mô hình tại {args.model}")
        return

    # 1. Tải và phân tích nhãn KITTI
    car_tracks, ped_tracks = parse_kitti_gt(args.label_dir, args.image_dir, max_sequences=args.max_seq)
    
    # 2. Đánh giá riêng cho Ô tô (Cars)
    res_car = evaluate_reid_on_tracks(car_tracks, "Car (Ô tô)", args.model, min_track_len=5, max_samples_per_track=8)
    
    # 3. Đánh giá riêng cho Người đi bộ (Pedestrians)
    res_ped = evaluate_reid_on_tracks(ped_tracks, "Pedestrian (Người đi bộ)", args.model, min_track_len=4, max_samples_per_track=8)
    
    # 4. In bảng tổng kết độ chính xác
    print("\n" + "="*75)
    print("📊 KẾT QUẢ ĐỘ CHÍNH XÁC CROSS-DOMAIN ReID TRÊN TẬP DỮ LIỆU KITTI (UNSEEN)")
    print("="*75)
    print(f"{'Loại đối tượng':<22} | {'Rank-1 (%)':<10} | {'Rank-5 (%)':<10} | {'Rank-10 (%)':<11} | {'mAP (%)':<10}")
    print("-" * 75)
    if res_car:
        print(f"{res_car['class']:<22} | {res_car['rank1']:<10.2f} | {res_car['rank5']:<10.2f} | {res_car['rank10']:<11.2f} | {res_car['mAP']:<10.2f}")
    if res_ped:
        print(f"{res_ped['class']:<22} | {res_ped['rank1']:<10.2f} | {res_ped['rank5']:<10.2f} | {res_ped['rank10']:<11.2f} | {res_ped['mAP']:<10.2f}")
    print("="*75)
    
    # 5. Kiểm tra phân tách Người vs Xe
    if res_car and res_ped:
        c_feats = res_car['sample_feats'][:min(100, len(res_car['sample_feats']))]
        p_feats = res_ped['sample_feats'][:min(100, len(res_ped['sample_feats']))]
        cross_cos = 1.0 - np.dot(p_feats, c_feats.T)
        mean_cross_dist = np.mean(cross_cos)
        min_cross_dist = np.min(cross_cos)
        print("\n🚗👤 KIỂM CHỨNG TÁCH BIỆT KHÔNG GIAN ĐẶC TRƯNG NGƯỜI VS XE:")
        print(f"  🔹 Khoảng cách Cosine trung bình giữa Người và Xe: {mean_cross_dist:.4f} (Tiêu chuẩn: > 0.70)")
        print(f"  🔹 Khoảng cách Cosine nhỏ nhất giữa Người và Xe   : {min_cross_dist:.4f}")
        if mean_cross_dist > 0.70:
            print("  ✅ [ĐẠT CHUẨN]: Hai không gian vector tách biệt rõ ràng, không có nguy cơ nhận nhầm Người thành Xe!")
            
    # 6. Đo đạc tốc độ suy luận trần (Pure Throughput & Latency)
    print("\n" + "="*75)
    print("⚡ ĐO ĐẠC TỐC ĐỘ VÀ ĐỘ TRỄ SUY LUẬN TRẦN TRÊN CPU (PURE REID BENCHMARK)")
    print("="*75)
    ie = Core()
    model = ie.read_model(args.model)
    compiled = ie.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    infer_req = compiled.create_infer_request()
    dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
    
    # Warmup
    for _ in range(50):
        infer_req.infer({0: dummy})
        
    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        infer_req.infer({0: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    lat_mean = np.mean(latencies)
    lat_p50 = np.percentile(latencies, 50)
    lat_p95 = np.percentile(latencies, 95)
    lat_p99 = np.percentile(latencies, 99)
    
    # Multi-stream Throughput
    compiled_tp = ie.compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
    queue_tp = AsyncInferQueue(compiled_tp)
    for _ in range(50):
        queue_tp.start_async({0: dummy})
    queue_tp.wait_all()
    
    N = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        queue_tp.start_async({0: dummy})
    queue_tp.wait_all()
    total_sec = time.perf_counter() - t0
    tp_crops_sec = N / total_sec
    
    print(f"  ⏱️ Độ trễ trung bình trên 1 Crop (Latency Mean) : {lat_mean:.2f} ms")
    print(f"  ⏱️ Độ trễ trung vị (P50)                         : {lat_p50:.2f} ms")
    print(f"  ⏱️ Độ trễ phân vị P95                            : {lat_p95:.2f} ms")
    print(f"  ⏱️ Độ trễ phân vị P99                            : {lat_p99:.2f} ms")
    print(f"  🚀 Băng thông suy luận đa luồng CPU (Throughput) : {tp_crops_sec:,.1f} crops/giây")
    print("="*75)
    print("🎉 Hoàn tất đánh giá độc lập!")

if __name__ == "__main__":
    main()
