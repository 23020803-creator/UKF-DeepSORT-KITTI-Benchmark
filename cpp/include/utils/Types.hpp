#pragma once

#include <vector>
#include <string>
#include <algorithm>

namespace ukf_deepsort {

/**
 * @brief Hộp bao 2D định dạng TLWH (Top-Left Width Height).
 */
struct BBox {
    float x{0.0f};  // Tọa độ x góc trên bên trái
    float y{0.0f};  // Tọa độ y góc trên bên trái
    float w{0.0f};  // Chiều rộng hộp bao
    float h{0.0f};  // Chiều cao hộp bao

    BBox() = default;
    BBox(float x_, float y_, float w_, float h_) : x(x_), y(y_), w(w_), h(h_) {}

    inline float x1() const { return x; }
    inline float y1() const { return y; }
    inline float x2() const { return x + w; }
    inline float y2() const { return y + h; }
    inline float area() const { return (w > 0.0f && h > 0.0f) ? (w * h) : 0.0f; }

    inline float cx() const { return x + w * 0.5f; }
    inline float cy() const { return y + h * 0.5f; }
    inline float aspect_ratio() const { return (h > 1e-4f) ? (w / h) : 1.0f; }
};

/**
 * @brief Cấu trúc dữ liệu đại diện cho một kết quả nhận diện (Detection) từ mạng YOLO.
 */
struct Detection {
    BBox bbox;
    float conf{0.0f};
    int class_id{0};                 // 0: Pedestrian, 2: Car
    std::vector<float> feature;      // Vector đặc trưng ngoại hình ReID (256 chiều)

    Detection() = default;
    Detection(const BBox& b, float c, int cls, const std::vector<float>& feat = {})
        : bbox(b), conf(c), class_id(cls), feature(feat) {}
};

/**
 * @brief Trạng thái vòng đời của một quỹ đạo (Track).
 */
enum class TrackState {
    TENTATIVE = 1,   // Trạng thái tạm thời, mới khởi tạo
    CONFIRMED = 2,   // Trạng thái đã xác nhận uy tín
    DELETED   = 3    // Trạng thái đã mất dấu quá lâu, chuẩn bị hủy
};

/**
 * @brief Kết quả theo dõi xuất ra file hoặc vẽ lên màn hình theo chuẩn MOT16.
 */
struct TrackResult {
    int frame_idx{0};
    int track_id{0};
    int class_id{0};
    BBox bbox;
    float conf{1.0f};

    TrackResult() = default;
    TrackResult(int frame, int id, int cls, const BBox& b, float c = 1.0f)
        : frame_idx(frame), track_id(id), class_id(cls), bbox(b), conf(c) {}
};

} // namespace ukf_deepsort
