#include "core/Tracker.hpp"
#include "core/Matching.hpp"
#include <algorithm>

namespace ukf_deepsort {

Tracker::Tracker(int max_age, int n_init, double cosine_threshold, double iou_threshold, double fps)
    : max_age_(max_age), n_init_(n_init), cosine_threshold_(cosine_threshold), iou_threshold_(iou_threshold), fps_(fps) {}

void Tracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

void Tracker::update(const std::vector<Detection>& detections, int img_w, int img_h, const cv::Mat& H_camera, int /*frame_idx*/) {
    // 1. Dự đoán tất cả các quỹ đạo bằng UKF (Bù trừ CMC)
    for (auto& track : tracks_) {
        track->predict(H_camera);

        // Kiểm tra điều kiện loại bỏ ở biên ảnh
        BBox b = track->to_tlwh();
        float cx = b.cx();
        float cy = b.cy();
        float margin = 10.0f;
        bool is_touching_margin = (b.x <= margin || b.y <= margin || 
                                   b.x + b.w >= img_w - margin || b.y + b.h >= img_h - margin);
        bool is_outside = (cx < 0.0f || cx > img_w || cy < 0.0f || cy > img_h);

        if ((is_touching_margin || is_outside) && track->time_since_update() > 15) {
            track->mark_missed();
        }
    }

    // 2. Lọc bớt nhiễu phát hiện ở sát lề nếu quá nhỏ
    std::vector<Detection> valid_dets;
    for (const auto& det : detections) {
        float margin_prune = 15.0f;
        bool is_edge = (det.bbox.x <= margin_prune || det.bbox.x + det.bbox.w >= img_w - margin_prune);
        if (is_edge && det.bbox.w < 60.0f) {
            continue;
        }
        valid_dets.push_back(det);
    }

    std::vector<int> unmatched_trks_idx;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!tracks_[i]->is_deleted()) {
            unmatched_trks_idx.push_back(static_cast<int>(i));
        }
    }
    std::vector<int> unmatched_dets_idx(valid_dets.size());
    for (size_t i = 0; i < valid_dets.size(); ++i) unmatched_dets_idx[i] = static_cast<int>(i);

    // =========================================================================
    // VÒNG 1: CASCADE REID MATCHING
    // =========================================================================
    for (int level = 1; level <= max_age_; ++level) {
        if (unmatched_dets_idx.empty()) break;

        std::vector<int> level_trk_idx;
        for (int idx : unmatched_trks_idx) {
            if (tracks_[idx]->time_since_update() == level) {
                level_trk_idx.push_back(idx);
            }
        }
        if (level_trk_idx.empty()) continue;

        std::vector<Track*> level_tracks;
        for (int idx : level_trk_idx) level_tracks.push_back(tracks_[idx].get());

        std::vector<Detection> level_dets;
        for (int idx : unmatched_dets_idx) level_dets.push_back(valid_dets[idx]);

        auto cost_matrix = matching::compute_cosine_distance(level_tracks, level_dets);
        auto maha_matrix = matching::compute_mahalanobis_distance(level_tracks, level_dets);

        // Áp dụng rào chắn không gian (Spatial Mahalanobis Gating)
        for (size_t r = 0; r < level_tracks.size(); ++r) {
            for (size_t c = 0; c < level_dets.size(); ++c) {
                if (level_tracks[r]->class_id() != level_dets[c].class_id || 
                    cost_matrix[r][c] > cosine_threshold_) 
                {
                    cost_matrix[r][c] = 1e5;
                    continue;
                }

                double sq_maha = maha_matrix[r][c];
                double chi2_gate = (level_tracks[r]->class_id() == 2) ? 25.0 : 18.0;
                if (sq_maha > chi2_gate) {
                    cost_matrix[r][c] = 1e5;
                }
            }
        }

        std::vector<std::pair<int, int>> matches_v1;
        std::vector<int> un_t, un_d;
        matching::linear_assignment(cost_matrix, level_tracks, level_dets, cosine_threshold_, matches_v1, un_t, un_d);

        for (const auto& match : matches_v1) {
            int actual_trk_idx = level_trk_idx[match.first];
            int actual_det_idx = unmatched_dets_idx[match.second];
            const auto& det = valid_dets[actual_det_idx];

            tracks_[actual_trk_idx]->update(det.bbox, det.feature, img_w, img_h);
            unmatched_trks_idx.erase(std::remove(unmatched_trks_idx.begin(), unmatched_trks_idx.end(), actual_trk_idx), unmatched_trks_idx.end());
        }

        std::vector<int> new_un_dets;
        for (int d_idx : un_d) new_un_dets.push_back(unmatched_dets_idx[d_idx]);
        unmatched_dets_idx = new_un_dets;
    }

    // =========================================================================
    // VÒNG 2: IOU MATCHING (Cho các track mới mất dấu: age == 1)
    // =========================================================================
    if (!unmatched_trks_idx.empty() && !unmatched_dets_idx.empty()) {
        std::vector<int> iou_trk_idx;
        for (int idx : unmatched_trks_idx) {
            if (tracks_[idx]->time_since_update() == 1) {
                iou_trk_idx.push_back(idx);
            }
        }

        if (!iou_trk_idx.empty()) {
            std::vector<Track*> iou_tracks;
            for (int idx : iou_trk_idx) iou_tracks.push_back(tracks_[idx].get());

            std::vector<Detection> iou_dets;
            for (int idx : unmatched_dets_idx) iou_dets.push_back(valid_dets[idx]);

            auto iou_cost = matching::compute_iou_matrix(iou_tracks, iou_dets, img_w, img_h);
            auto feature_cost_v2 = matching::compute_cosine_distance(iou_tracks, iou_dets);

            for (size_t r = 0; r < iou_tracks.size(); ++r) {
                for (size_t c = 0; c < iou_dets.size(); ++c) {
                    if (iou_tracks[r]->class_id() != iou_dets[c].class_id || feature_cost_v2[r][c] > 0.55) {
                        iou_cost[r][c] = 1e5;
                    }
                }
            }

            std::vector<std::pair<int, int>> matches_v2;
            std::vector<int> un_t_iou, un_d_iou;
            matching::linear_assignment(iou_cost, iou_tracks, iou_dets, iou_threshold_, matches_v2, un_t_iou, un_d_iou);

            for (const auto& match : matches_v2) {
                int actual_trk_idx = iou_trk_idx[match.first];
                int actual_det_idx = unmatched_dets_idx[match.second];
                const auto& det = valid_dets[actual_det_idx];

                tracks_[actual_trk_idx]->update(det.bbox, det.feature, img_w, img_h);
                unmatched_trks_idx.erase(std::remove(unmatched_trks_idx.begin(), unmatched_trks_idx.end(), actual_trk_idx), unmatched_trks_idx.end());
            }

            std::vector<int> new_un_dets;
            for (int d_idx : un_d_iou) new_un_dets.push_back(unmatched_dets_idx[d_idx]);
            unmatched_dets_idx = new_un_dets;
        }
    }

    // =========================================================================
    // VÒNG 3: RECOVERY IOM MATCHING (Cứu nguy các track còn lại)
    // =========================================================================
    if (!unmatched_trks_idx.empty() && !unmatched_dets_idx.empty()) {
        std::vector<Track*> rec_tracks;
        for (int idx : unmatched_trks_idx) rec_tracks.push_back(tracks_[idx].get());

        std::vector<Detection> rec_dets;
        for (int idx : unmatched_dets_idx) rec_dets.push_back(valid_dets[idx]);

        auto bypass_cost = matching::compute_iou_matrix(rec_tracks, rec_dets, img_w, img_h);

        std::vector<std::pair<int, int>> matches_v3;
        std::vector<int> un_t_v3, un_d_v3;
        matching::linear_assignment(bypass_cost, rec_tracks, rec_dets, 1.0, matches_v3, un_t_v3, un_d_v3);

        for (const auto& match : matches_v3) {
            int actual_trk_idx = unmatched_trks_idx[match.first];
            int actual_det_idx = unmatched_dets_idx[match.second];
            const auto& det = valid_dets[actual_det_idx];

            tracks_[actual_trk_idx]->update(det.bbox, det.feature, img_w, img_h);
            unmatched_trks_idx.erase(std::remove(unmatched_trks_idx.begin(), unmatched_trks_idx.end(), actual_trk_idx), unmatched_trks_idx.end());
        }

        std::vector<int> new_un_dets;
        for (int d_idx : un_d_v3) new_un_dets.push_back(unmatched_dets_idx[d_idx]);
        unmatched_dets_idx = new_un_dets;
    }

    // =========================================================================
    // QUẢN LÝ VÒNG ĐỜI (Lifecycle Management)
    // =========================================================================
    // Khởi tạo track mới cho các detection chưa ghép
    for (int det_idx : unmatched_dets_idx) {
        const auto& det = valid_dets[det_idx];
        if (det.conf > 0.5f) {
            tracks_.push_back(std::make_unique<Track>(fps_, next_id_++, det.class_id, det.bbox, det.feature));
        }
    }

    // Đánh dấu missed cho các track chưa ghép
    for (int trk_idx : unmatched_trks_idx) {
        tracks_[trk_idx]->mark_missed();
    }

    // Xóa bỏ các track đã DELETED
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(), 
            [](const std::unique_ptr<Track>& t) { return t->is_deleted(); }),
        tracks_.end()
    );
}

std::vector<TrackResult> Tracker::get_results(bool only_detected, int frame_idx) const {
    std::vector<TrackResult> results;
    for (const auto& track : tracks_) {
        if (!track->is_confirmed()) continue;
        if (only_detected && track->time_since_update() > 0) continue;

        BBox b = track->to_tlwh();
        results.emplace_back(frame_idx, track->track_id(), track->class_id(), b, 1.0f);
    }
    return results;
}

} // namespace ukf_deepsort
