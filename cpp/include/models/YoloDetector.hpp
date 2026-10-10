#pragma once

#include "models/TrtEngine.hpp"
#include "utils/Types.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <memory>

namespace ukf_deepsort {

/**
 * @brief Lớp nhận diện vật thể YOLO tăng tốc bằng TensorRT trên NVIDIA Jetson Nano.
 */
class YoloDetector {
public:
    explicit YoloDetector(const std::string& engine_path, float conf_thresh = 0.55f, float nms_thresh = 0.45f);

    /**
     * @brief Thực hiện phát hiện vật thể trên ảnh gốc BGR.
     * @param image Ảnh đầu vào (BGR).
     * @return std::vector<Detection> Danh sách các đối tượng phát hiện được (Class 0: Người, Class 2: Xe).
     */
    std::vector<Detection> detect(const cv::Mat& image);

    inline bool is_valid() const { return engine_ && engine_->is_valid(); }

private:
    std::unique_ptr<TrtEngine> engine_;
    float conf_thresh_{0.55f};
    float nms_thresh_{0.45f};

    int input_w_{640};
    int input_h_{640};

    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;

    void preprocess(const cv::Mat& image, float& scale, int& pad_x, int& pad_y);
    std::vector<Detection> postprocess(const cv::Mat& image, float scale, int pad_x, int pad_y);
    static void nms(std::vector<Detection>& dets, float nms_thresh);
};

} // namespace ukf_deepsort
