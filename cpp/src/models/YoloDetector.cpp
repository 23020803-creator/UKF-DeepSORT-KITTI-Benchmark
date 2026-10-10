#include "models/YoloDetector.hpp"
#include <algorithm>
#include <iostream>

namespace ukf_deepsort {

YoloDetector::YoloDetector(const std::string& engine_path, float conf_thresh, float nms_thresh)
    : conf_thresh_(conf_thresh), nms_thresh_(nms_thresh) {
    engine_ = std::make_unique<TrtEngine>(engine_path);
    input_buffer_.resize(1 * 3 * input_h_ * input_w_, 0.0f);
    // YOLOv8/11/26 output: [1, 84, 8400]
    output_buffer_.resize(1 * 84 * 8400, 0.0f);
}

void YoloDetector::preprocess(const cv::Mat& image, float& scale, int& pad_x, int& pad_y) {
    float r = std::min(static_cast<float>(input_w_) / image.cols, static_cast<float>(input_h_) / image.rows);
    int new_unpad_w = static_cast<int>(std::round(image.cols * r));
    int new_unpad_h = static_cast<int>(std::round(image.rows * r));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));

    pad_x = (input_w_ - new_unpad_w) / 2;
    pad_y = (input_h_ - new_unpad_h) / 2;

    cv::Mat padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_unpad_w, new_unpad_h)));

    scale = r;

    // Chuyển BGR -> RGB, chuẩn hóa [0, 1] và định dạng CHW
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    int channel_size = input_w_ * input_h_;
    float* r_plane = input_buffer_.data();
    float* g_plane = input_buffer_.data() + channel_size;
    float* b_plane = input_buffer_.data() + 2 * channel_size;

    for (int y = 0; y < input_h_; ++y) {
        const cv::Vec3b* ptr = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_w_; ++x) {
            int idx = y * input_w_ + x;
            r_plane[idx] = ptr[x][0] / 255.0f;
            g_plane[idx] = ptr[x][1] / 255.0f;
            b_plane[idx] = ptr[x][2] / 255.0f;
        }
    }
}

void YoloDetector::nms(std::vector<Detection>& dets, float nms_thresh) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });

    std::vector<Detection> keep;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);

        const BBox& b1 = dets[i].bbox;
        float a1 = b1.area();

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue;

            const BBox& b2 = dets[j].bbox;
            float x1 = std::max(b1.x, b2.x);
            float y1 = std::max(b1.y, b2.y);
            float x2 = std::min(b1.x + b1.w, b2.x + b2.w);
            float y2 = std::min(b1.y + b1.h, b2.y + b2.h);

            if (x2 > x1 && y2 > y1) {
                float inter = (x2 - x1) * (y2 - y1);
                float iou = inter / (a1 + b2.area() - inter + 1e-6f);
                if (iou > nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    dets = keep;
}

std::vector<Detection> YoloDetector::postprocess(const cv::Mat& image, float scale, int pad_x, int pad_y) {
    std::vector<Detection> raw_dets;

    // Output shape [1, 84, 8400]: 84 channels (4 box coords + 80 classes) x 8400 anchors
    constexpr int NUM_CHANNELS = 84;
    constexpr int NUM_ANCHORS = 8400;

    for (int i = 0; i < NUM_ANCHORS; ++i) {
        // Tìm class có confidence lớn nhất
        float max_score = 0.0f;
        int max_cls = -1;

        for (int c = 0; c < 80; ++c) {
            float score = output_buffer_[(4 + c) * NUM_ANCHORS + i];
            if (score > max_score) {
                max_score = score;
                max_cls = c;
            }
        }

        // Chỉ lọc Person (Class 0) và Car (Class 2)
        if ((max_cls == 0 || max_cls == 2) && max_score >= conf_thresh_) {
            float cx = output_buffer_[0 * NUM_ANCHORS + i];
            float cy = output_buffer_[1 * NUM_ANCHORS + i];
            float w  = output_buffer_[2 * NUM_ANCHORS + i];
            float h  = output_buffer_[3 * NUM_ANCHORS + i];

            // Chuyển ngược từ padded sang ảnh gốc
            float x1 = (cx - w * 0.5f - pad_x) / scale;
            float y1 = (cy - h * 0.5f - pad_y) / scale;
            float orig_w = w / scale;
            float orig_h = h / scale;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows - 1)));
            orig_w = std::min(orig_w, static_cast<float>(image.cols) - x1);
            orig_h = std::min(orig_h, static_cast<float>(image.rows) - y1);

            raw_dets.emplace_back(BBox(x1, y1, orig_w, orig_h), max_score, max_cls);
        }
    }

    nms(raw_dets, nms_thresh_);
    return raw_dets;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& image) {
    if (image.empty() || !is_valid()) return {};

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    preprocess(image, scale, pad_x, pad_y);

    std::vector<void*> host_inputs = { input_buffer_.data() };
    std::vector<void*> host_outputs = { output_buffer_.data() };

    if (!engine_->infer(host_inputs, host_outputs)) {
        return {};
    }

    return postprocess(image, scale, pad_x, pad_y);
}

} // namespace ukf_deepsort
