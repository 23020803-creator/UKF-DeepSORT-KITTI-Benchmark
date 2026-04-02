import time
import numpy as np
import cv2

from utils.kitti_parser import KittiParser
from utils.visualizer import Visualizer
from models.yolo_detector import YOLODetector
from models.reid_extractor import ReIDExtractor
from core import matching

class Detection:
    def __init__(self, bbox, conf, class_id, feature=None):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.conf = float(conf)
        self.class_id = int(class_id)
        self.feature = feature

class SimpleTrack:
    def __init__(self, track_id, class_id, bbox, feature):
        self.track_id = track_id
        self.class_id = int(class_id)
        self.bbox = bbox 
        self.feature = feature
        self.time_since_update = 0

def compute_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    cost_matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
            x_right, y_bottom = min(box1[0] + box1[2], box2[0] + box2[2]), min(box1[1] + box1[3], box2[1] + box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                iou = 0.0
            else:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1, area2 = box1[2]*box1[3], box2[2]*box2[3]
                iou = intersection / float(area1 + area2 - intersection + 1e-6)
            cost_matrix[i, j] = 1.0 - iou
    return cost_matrix

def get_danger_zones(bboxes, iou_threshold=0.15):
    num_boxes = len(bboxes)
    is_danger = np.zeros(num_boxes, dtype=bool)
    if num_boxes < 2:
        return is_danger
        
    cost_matrix = compute_iou_matrix(bboxes, bboxes)
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if cost_matrix[i, j] < (1.0 - iou_threshold):
                is_danger[i] = True
                is_danger[j] = True
    return is_danger

def main():
    print("=== PURE REID TRACKING: OCCLUSION-AWARE CASCADE (FIXED) ===")
    
    parser = KittiParser(seq_dir="datasets/KITTI_MOT/KITTI-0000")
    visualizer = Visualizer(output_path="outputs/videos/test_occlusion_aware.mp4", fps=40)
    
    detector = YOLODetector(model_path="weights/yolo11n_int8_openvino_model", conf_thresh=0.55)
    extractor = ReIDExtractor(model_path="weights/public/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.xml", device="CPU")
    
    tracks = [] 
    next_id = 1
    
    for frame_idx, image, _ in parser.get_frame():
        frame_start_time = time.time()
        
        bboxes, confs, class_ids = detector.detect(image)
        valid_dets = [Detection(b, c, cls_id) for b, c, cls_id in zip(bboxes, confs, class_ids)]
        
        t_reid_start = time.time()
        
        if len(valid_dets) > 0:
            det_bboxes = np.array([d.bbox for d in valid_dets])
            is_danger = get_danger_zones(det_bboxes, iou_threshold=0.15)
            safe_det_indices = np.where(~is_danger)[0].tolist()
            danger_det_indices = np.where(is_danger)[0].tolist()
        else:
            safe_det_indices, danger_det_indices = [], []

        unmatched_trks = list(range(len(tracks)))
        new_active_tracks = []
        
        # =========================================================
        # LUỒNG 1: IOU CHO VÙNG AN TOÀN
        # =========================================================
        if len(safe_det_indices) > 0:
            if len(unmatched_trks) > 0: 
                safe_dets = [valid_dets[i] for i in safe_det_indices]
                trk_bboxes = [tracks[i].bbox for i in unmatched_trks]
                safe_bboxes = [d.bbox for d in safe_dets]
                
                iou_cost = compute_iou_matrix(trk_bboxes, safe_bboxes)
                
                matches_iou, un_trk_idx, un_det_idx = matching.linear_assignment(
                    cost_matrix=iou_cost, tracks=[tracks[i] for i in unmatched_trks], detections=safe_dets, max_distance=0.6 
                )
                
                for trk_i, safe_det_i in matches_iou:
                    trk = tracks[unmatched_trks[trk_i]]
                    trk.bbox = safe_dets[safe_det_i].bbox
                    trk.time_since_update = 0
                    new_active_tracks.append(trk)
                    
                unmatched_trks = [unmatched_trks[i] for i in un_trk_idx]
                danger_det_indices.extend([safe_det_indices[i] for i in un_det_idx])
            else:
                danger_det_indices.extend(safe_det_indices)

        # =========================================================
        # LUỒNG 2: REID CHO VÙNG NGUY HIỂM / XE MỚI
        # =========================================================
        if len(danger_det_indices) > 0:
            danger_dets = [valid_dets[i] for i in danger_det_indices]
            danger_bboxes = np.array([d.bbox for d in danger_dets])
            
            danger_features = extractor.extract(image, danger_bboxes)
            for i, d in enumerate(danger_dets):
                d.feature = danger_features[i]
            
            if len(unmatched_trks) > 0:
                trk_features = np.array([tracks[i].feature for i in unmatched_trks])
                cosine_cost = matching.compute_cosine_distance(trk_features, danger_features)
                
                matches_reid, un_trk_idx, un_det_idx = matching.linear_assignment(
                    cost_matrix=cosine_cost, tracks=[tracks[i] for i in unmatched_trks], detections=danger_dets, max_distance=0.35
                )
                
                for trk_i, danger_det_i in matches_reid:
                    trk = tracks[unmatched_trks[trk_i]]
                    det = danger_dets[danger_det_i]
                    
                    trk.bbox = det.bbox
                    trk.feature = 0.9 * trk.feature + 0.1 * det.feature
                    trk.feature /= max(np.linalg.norm(trk.feature), 1e-6)
                    trk.time_since_update = 0
                    new_active_tracks.append(trk)
                    
                unmatched_trks = [unmatched_trks[i] for i in un_trk_idx]
                unmatched_danger_dets = [danger_dets[i] for i in un_det_idx]
            else:
                unmatched_danger_dets = danger_dets
                
            for det in unmatched_danger_dets:
                new_track = SimpleTrack(track_id=next_id, class_id=det.class_id, bbox=det.bbox, feature=det.feature)
                new_active_tracks.append(new_track)
                next_id += 1

        for trk_i in unmatched_trks:
            trk = tracks[trk_i]
            trk.time_since_update += 1
            if trk.time_since_update <= 30:
                new_active_tracks.append(trk)

        tracks = new_active_tracks
        
        time_reid_total = (time.time() - t_reid_start) * 1000
        total_time = time.time() - frame_start_time
        fps = 1.0 / total_time
        
        draw_data = [(t.track_id, t.class_id, t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]) for t in tracks if t.time_since_update == 0]
        drawn_img = visualizer.draw_tracks(image, draw_data, frame_idx)
        
        print(f"Frame: {frame_idx:04d} | Xe: {len(draw_data):02d} | Cảnh báo: {len(danger_det_indices)} xe | ReID: {time_reid_total:4.1f}ms | FPS: {fps:4.1f}")
        
        # ĐÃ BẬT LẠI SHOW VIDEO TRỰC TIẾP ĐỂ BẠN CHECK BOX
        cv2.imshow("Test Pipeline", drawn_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    visualizer.release()
    cv2.destroyAllWindows()
    print("=== HOÀN THÀNH BÀI TEST ===")

if __name__ == "__main__":
    main()