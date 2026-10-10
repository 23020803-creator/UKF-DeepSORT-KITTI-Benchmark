#pragma once

#include "models/TrtEngine.hpp"
#include "utils/Types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <memory>

namespace ukf_deepsort {

/**
 * @brief Lớp trích xuất đặc trưng nhận dạng ReID (Unified ReID 256 chiều) bằng TensorRT.
 */
class ReidExtractor {
public:
    explicit ReidExtractor(const std::string& engine_path);

    /**
     * @brief Trích xuất vector đặc trưng ReID cho danh sách các bounding boxes trong một khung hình.
     * @param image Ảnh gốc BGR.
     * @param detections Danh sách phát hiện (hàm sẽ điền trực tiếp vector đặc trưng vào trường feature).
     */
    void extract(const cv::Mat& image, std::vector<Detection>& detections);

    /**
     * @brief Trích xuất vector đặc trưng cho một crop duy nhất.
     */
    std::vector<float> extract_crop(const cv::Mat& crop);

    inline bool is_valid() const { return engine_ && engine_->is_valid(); }

private:
    std::unique_ptr<TrtEngine> engine_;

    int input_w_{128};
    int input_h_{128};
    int feat_dim_{256};

    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;

    void preprocess_crop(const cv::Mat& crop);
};

} // namespace ukf_deepsort
