#pragma once

#include "core/Track.hpp"
#include "utils/Types.hpp"
#include <vector>
#include <memory>

namespace ukf_deepsort {

/**
 * @brief Bộ điều phối theo dõi đa mục tiêu (Tracker) thực hiện quy trình 3 vòng ghép cặp cascade.
 */
class Tracker {
public:
    Tracker(int max_age = 30, int n_init = 3, double cosine_threshold = 0.42, double iou_threshold = 0.7, double fps = 10.0);

    /**
     * @brief Cập nhật trạng thái Tracker cho khung hình mới.
     * @param detections Danh sách phát hiện YOLO kèm đặc trưng ReID.
     * @param img_w Chiều rộng khung hình.
     * @param img_h Chiều cao khung hình.
     * @param H_camera Ma trận Affine 2x3 từ module CMC.
     * @param frame_idx Chỉ số khung hình hiện tại.
     */
    void update(const std::vector<Detection>& detections, int img_w, int img_h, const cv::Mat& H_camera = cv::Mat(), int frame_idx = 0);

    /**
     * @brief Lấy kết quả các quỹ đạo để xuất ra file hoặc vẽ hiển thị.
     * @param only_detected Chỉ lấy các track vừa được cập nhật detection ở frame này (chuẩn Benchmark).
     */
    std::vector<TrackResult> get_results(bool only_detected = true, int frame_idx = 0) const;

    /**
     * @brief Đặt lại bộ nhớ khi chuyển sequence video mới.
     */
    void reset();

    inline const std::vector<std::unique_ptr<Track>>& tracks() const { return tracks_; }

private:
    int max_age_{30};
    int n_init_{3};
    double cosine_threshold_{0.42};
    double iou_threshold_{0.7};
    double fps_{10.0};
    int next_id_{1};

    std::vector<std::unique_ptr<Track>> tracks_;
};

} // namespace ukf_deepsort
