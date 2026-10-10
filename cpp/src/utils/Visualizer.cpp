#include "utils/Visualizer.hpp"
#include <string>

namespace ukf_deepsort {

Visualizer::Visualizer() {
    // Khởi tạo bảng màu sắc phân biệt các ID
    color_palette_ = {
        cv::Scalar(255, 56, 56),   cv::Scalar(255, 157, 151), cv::Scalar(255, 112, 31),
        cv::Scalar(255, 178, 29),  cv::Scalar(207, 210, 49),  cv::Scalar(72, 249, 10),
        cv::Scalar(146, 204, 23),  cv::Scalar(61, 219, 134),  cv::Scalar(26, 147, 52),
        cv::Scalar(0, 212, 187),   cv::Scalar(44, 153, 168),  cv::Scalar(0, 194, 255),
        cv::Scalar(52, 69, 147),   cv::Scalar(100, 115, 255), cv::Scalar(0, 24, 236),
        cv::Scalar(132, 56, 255),  cv::Scalar(82, 0, 133),    cv::Scalar(203, 56, 255),
        cv::Scalar(255, 149, 200), cv::Scalar(255, 55, 199)
    };
}

cv::Scalar Visualizer::get_color(int id) const {
    if (color_palette_.empty()) return cv::Scalar(0, 255, 0);
    return color_palette_[static_cast<size_t>(id) % color_palette_.size()];
}

void Visualizer::draw(cv::Mat& image, const std::vector<TrackResult>& results, double fps) {
    if (image.empty()) return;

    for (const auto& res : results) {
        cv::Scalar color = get_color(res.track_id);
        cv::Rect rect(static_cast<int>(res.bbox.x), static_cast<int>(res.bbox.y),
                      static_cast<int>(res.bbox.w), static_cast<int>(res.bbox.h));

        // Vẽ bounding box
        cv::rectangle(image, rect, color, 2);

        // Lưu và vẽ đuôi quỹ đạo di chuyển
        cv::Point center(static_cast<int>(res.bbox.cx()), static_cast<int>(res.bbox.cy()));
        auto& pts = trajectories_[res.track_id];
        pts.push_back(center);
        if (pts.size() > 30) pts.pop_front();

        for (size_t i = 1; i < pts.size(); ++i) {
            cv::line(image, pts[i - 1], pts[i], color, 2);
        }

        // Vẽ nhãn văn bản
        std::string cls_str = (res.class_id == 2) ? "Car" : "Ped";
        std::string label = "ID:" + std::to_string(res.track_id) + " " + cls_str;

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(rect.y, label_size.height);

        cv::rectangle(image, cv::Point(rect.x, top - label_size.height),
                             cv::Point(rect.x + label_size.width, top + baseLine), color, cv::FILLED);
        cv::putText(image, label, cv::Point(rect.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // Hiển thị FPS
    if (fps > 0.0) {
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(image, fps_text, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
}

} // namespace ukf_deepsort
