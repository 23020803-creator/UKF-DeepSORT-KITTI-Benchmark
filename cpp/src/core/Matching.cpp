#include "core/Matching.hpp"
#include "third_party/Hungarian.hpp"
#include <cmath>
#include <algorithm>
#include <set>

namespace ukf_deepsort {
namespace matching {

std::vector<std::vector<double>> compute_cosine_distance(
    const std::vector<Track*>& tracks, 
    const std::vector<Detection>& detections) 
{
    size_t m = tracks.size();
    size_t n = detections.size();
    if (m == 0 || n == 0) return {};

    std::vector<std::vector<double>> cost_matrix(m, std::vector<double>(n, 1.0));

    for (size_t i = 0; i < m; ++i) {
        const auto& feat_t = tracks[i]->smooth_feature();
        if (feat_t.empty()) continue;

        for (size_t j = 0; j < n; ++j) {
            const auto& feat_d = detections[j].feature;
            if (feat_d.empty()) continue;

            double dot = 0.0;
            size_t len = std::min(feat_t.size(), feat_d.size());
            for (size_t k = 0; k < len; ++k) {
                dot += feat_t[k] * feat_d[k];
            }
            // Dot product của 2 vector đã L2-normalized chính là Cosine similarity
            double dist = 1.0 - dot;
            cost_matrix[i][j] = std::max(0.0, std::min(dist, 2.0));
        }
    }

    return cost_matrix;
}

std::vector<std::vector<double>> compute_iou_matrix(
    const std::vector<Track*>& tracks, 
    const std::vector<Detection>& detections, 
    int /*img_w*/, 
    int /*img_h*/) 
{
    size_t m = tracks.size();
    size_t n = detections.size();
    if (m == 0 || n == 0) return {};

    std::vector<std::vector<double>> cost_matrix(m, std::vector<double>(n, 1.0));

    for (size_t i = 0; i < m; ++i) {
        BBox box1 = tracks[i]->to_tlwh();
        for (size_t j = 0; j < n; ++j) {
            const BBox& box2 = detections[j].bbox;

            float x_left   = std::max(box1.x, box2.x);
            float y_top    = std::max(box1.y, box2.y);
            float x_right  = std::min(box1.x + box1.w, box2.x + box2.w);
            float y_bottom = std::min(box1.y + box1.h, box2.y + box2.h);

            if (x_right < x_left || y_bottom < y_top) {
                cost_matrix[i][j] = 1.0;
                continue;
            }

            float intersection = (x_right - x_left) * (y_bottom - y_top);
            float area1 = box1.area();
            float area2 = box2.area();

            float iou = intersection / (area1 + area2 - intersection + 1e-6f);
            float min_area = std::min(area1, area2);
            float iom = intersection / (min_area + 1e-6f);

            float best_metric = (iom > 0.8f) ? std::max(iou, iom) : iou;
            cost_matrix[i][j] = 1.0 - static_cast<double>(best_metric);
        }
    }

    return cost_matrix;
}

std::vector<std::vector<double>> compute_mahalanobis_distance(
    const std::vector<Track*>& tracks, 
    const std::vector<Detection>& detections) 
{
    size_t m = tracks.size();
    size_t n = detections.size();
    if (m == 0 || n == 0) return {};

    std::vector<std::vector<double>> dist_matrix(m, std::vector<double>(n, 1e5));

    for (size_t i = 0; i < m; ++i) {
        const auto& raw_state = tracks[i]->ukf().mean();
        double cx = raw_state(0), cy = raw_state(1), a = raw_state(2), h = raw_state(3);
        double w = a * h;

        double xmin = std::max(0.0, cx - w * 0.5);
        double xmax = std::min(1242.0, cx + w * 0.5);
        double ymin = std::max(0.0, cy - h * 0.5);
        double ymax = std::min(375.0, cy + h * 0.5);

        if (xmin >= xmax || ymin >= ymax) {
            continue;
        }

        double w_obs = xmax - xmin;
        double h_obs = ymax - ymin;
        double cx_obs = (xmin + xmax) * 0.5;
        double cy_obs = (ymin + ymax) * 0.5;
        double a_obs = (h_obs > 0.0) ? (w_obs / h_obs) : a;

        Eigen::Vector4d mean_proj;
        mean_proj << cx_obs, cy_obs, a_obs, h_obs;

        Eigen::Matrix4d P_proj = tracks[i]->ukf().covariance().block<4, 4>(0, 0);

        double std_pos = 0.02 * std::max(h, 1.0);
        Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
        R(0, 0) = std_pos * std_pos;
        R(1, 1) = std_pos * std_pos;
        R(2, 2) = 1e-4;
        R(3, 3) = std_pos * std_pos;

        Eigen::Matrix4d S = P_proj + R;
        Eigen::LLT<Eigen::Matrix4d> llt(S);
        if (llt.info() != Eigen::Success) {
            S += Eigen::Matrix4d::Identity() * 1e-6;
            llt.compute(S);
        }

        for (size_t j = 0; j < n; ++j) {
            auto z = Track::tlwh_to_xyah(detections[j].bbox);
            Eigen::Vector4d innovation = z - mean_proj;

            Eigen::Vector4d y = llt.solve(innovation);
            double sq_maha = innovation.dot(y);
            dist_matrix[i][j] = sq_maha;
        }
    }

    return dist_matrix;
}

void linear_assignment(
    std::vector<std::vector<double>>& cost_matrix,
    const std::vector<Track*>& tracks,
    const std::vector<Detection>& detections,
    double max_distance,
    std::vector<std::pair<int, int>>& matches,
    std::vector<int>& unmatched_tracks,
    std::vector<int>& unmatched_detections) 
{
    matches.clear();
    unmatched_tracks.clear();
    unmatched_detections.clear();

    if (cost_matrix.empty() || tracks.empty() || detections.empty()) {
        for (size_t i = 0; i < tracks.size(); ++i) unmatched_tracks.push_back(static_cast<int>(i));
        for (size_t j = 0; j < detections.size(); ++j) unmatched_detections.push_back(static_cast<int>(j));
        return;
    }

    size_t m = tracks.size();
    size_t n = detections.size();

    // Bước 1: Gating theo class ID
    constexpr double INFTY_COST = 1e5;
    for (size_t r = 0; r < m; ++r) {
        for (size_t c = 0; c < n; ++c) {
            if (tracks[r]->class_id() != detections[c].class_id) {
                cost_matrix[r][c] = INFTY_COST;
            }
        }
    }

    // Bước 2: Chạy Hungarian
    std::vector<int> assignment;
    HungarianAlgorithm::Solve(cost_matrix, assignment);

    std::set<int> assigned_dets;
    for (size_t r = 0; r < m; ++r) {
        int c = assignment[r];
        if (c >= 0 && cost_matrix[r][c] <= max_distance) {
            matches.emplace_back(static_cast<int>(r), c);
            assigned_dets.insert(c);
        } else {
            unmatched_tracks.push_back(static_cast<int>(r));
        }
    }

    for (size_t c = 0; c < n; ++c) {
        if (assigned_dets.find(static_cast<int>(c)) == assigned_dets.end()) {
            unmatched_detections.push_back(static_cast<int>(c));
        }
    }
}

} // namespace matching
} // namespace ukf_deepsort
