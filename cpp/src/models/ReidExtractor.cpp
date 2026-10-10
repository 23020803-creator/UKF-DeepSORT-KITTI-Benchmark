#include "models/ReidExtractor.hpp"
#include <cmath>
#include <algorithm>

namespace ukf_deepsort {

ReidExtractor::ReidExtractor(const std::string& engine_path) {
    engine_ = std::make_unique<TrtEngine>(engine_path);
    input_buffer_.resize(1 * 3 * input_h_ * input_w_, 0.0f);
    output_buffer_.resize(1 * feat_dim_, 0.0f);
}

void ReidExtractor::preprocess_crop(const cv::Mat& crop) {
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(input_w_, input_h_));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Chuẩn hóa ImageNet Z-score
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float std_dev[3] = {0.229f, 0.224f, 0.225f};

    int channel_size = input_w_ * input_h_;
    float* r_plane = input_buffer_.data();
    float* g_plane = input_buffer_.data() + channel_size;
    float* b_plane = input_buffer_.data() + 2 * channel_size;

    for (int y = 0; y < input_h_; ++y) {
        const cv::Vec3b* ptr = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w_; ++x) {
            int idx = y * input_w_ + x;
            r_plane[idx] = (ptr[x][0] / 255.0f - mean[0]) / std_dev[0];
            g_plane[idx] = (ptr[x][1] / 255.0f - mean[1]) / std_dev[1];
            b_plane[idx] = (ptr[x][2] / 255.0f - mean[2]) / std_dev[2];
        }
    }
}

std::vector<float> ReidExtractor::extract_crop(const cv::Mat& crop) {
    if (crop.empty() || !is_valid()) {
        return std::vector<float>(feat_dim_, 0.0f);
    }

    preprocess_crop(crop);

    std::vector<void*> host_inputs = { input_buffer_.data() };
    std::vector<void*> host_outputs = { output_buffer_.data() };

    if (!engine_->infer(host_inputs, host_outputs)) {
        return std::vector<float>(feat_dim_, 0.0f);
    }

    std::vector<float> feat = output_buffer_;

    // Chuẩn hóa L2-norm
    double sum_sq = 0.0;
    for (float v : feat) sum_sq += v * v;
    double norm = std::sqrt(sum_sq);
    if (norm > 1e-6) {
        for (float& v : feat) v /= static_cast<float>(norm);
    }

    return feat;
}

void ReidExtractor::extract(const cv::Mat& image, std::vector<Detection>& detections) {
    if (image.empty() || detections.empty()) return;

    for (auto& det : detections) {
        int x1 = std::max(0, static_cast<int>(det.bbox.x));
        int y1 = std::max(0, static_cast<int>(det.bbox.y));
        int w  = std::min(image.cols - x1, static_cast<int>(det.bbox.w));
        int h  = std::min(image.rows - y1, static_cast<int>(det.bbox.h));

        if (w <= 0 || h <= 0) {
            det.feature = std::vector<float>(feat_dim_, 0.0f);
            continue;
        }

        cv::Mat crop = image(cv::Rect(x1, y1, w, h));
        det.feature = extract_crop(crop);
    }
}

} // namespace ukf_deepsort
