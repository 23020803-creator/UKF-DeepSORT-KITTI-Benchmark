#pragma once

#include "utils/Types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <deque>

namespace ukf_deepsort {

/**
 * @brief Lớp hiển thị và vẽ kết quả theo dõi lên khung hình OpenCV.
 */
class Visualizer {
public:
    Visualizer();

    /**
     * @brief Vẽ các bounding boxes và quỹ đạo lên ảnh.
     * @param image Khung hình gốc (sẽ được vẽ đè lên).
     * @param results Danh sách các track hiện hành.
     * @param fps Tốc độ khung hình (FPS) hiện tại để hiển thị.
     */
    void draw(cv::Mat& image, const std::vector<TrackResult>& results, double fps = -1.0);

private:
    std::map<int, std::deque<cv::Point>> trajectories_;
    std::vector<cv::Scalar> color_palette_;

    cv::Scalar get_color(int id) const;
};

} // namespace ukf_deepsort
