#pragma once

#include "core/Track.hpp"
#include "utils/Types.hpp"
#include <vector>
#include <utility>

namespace ukf_deepsort {

namespace matching {

/**
 * @brief Tính ma trận khoảng cách Cosine giữa các quỹ đạo hiện tại và các đặc trưng nhận diện mới.
 */
std::vector<std::vector<double>> compute_cosine_distance(
    const std::vector<Track*>& tracks, 
    const std::vector<Detection>& detections
);

/**
 * @brief Tính ma trận chi phí IoU (kèm luật Asymmetrical IoM) giữa hộp UKF dự đoán và hộp YOLO.
 */
std::vector<std::vector<double>> compute_iou_matrix(
    const std::vector<Track*>& tracks, 
    const std::vector<Detection>& detections, 
    int img_w, 
    int img_h
);

/**
 * @brief Tính ma trận bình phương khoảng cách Mahalanobis (D^2) có tính đến hiệp phương sai UKF.
 */
std::vector<std::vector<double>> compute_mahalanobis_distance(
    const std::vector<Track*>& tracks, 
    const std::vector<Detection>& detections
);

/**
 * @brief Ghép cặp hai phía bằng thuật toán Hungarian với ngưỡng chi phí tối đa (Gating).
 */
void linear_assignment(
    std::vector<std::vector<double>>& cost_matrix,
    const std::vector<Track*>& tracks,
    const std::vector<Detection>& detections,
    double max_distance,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_detections
);

} // namespace matching

} // namespace ukf_deepsort
