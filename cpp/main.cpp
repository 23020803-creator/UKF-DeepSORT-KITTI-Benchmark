#include "core/CMC.hpp"
#include "core/Tracker.hpp"
#include "models/YoloDetector.hpp"
#include "models/ReidExtractor.hpp"
#include "utils/Visualizer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>

using namespace ukf_deepsort;

int main(int argc, char** argv) {
    std::cout << "========================================================" << std::endl;
    std::cout << "   UKF-DeepSORT C++ Deployment for NVIDIA Jetson Nano   " << std::endl;
    std::cout << "========================================================" << std::endl;

    std::string yolo_path = "weights/yolo26n_fp16.engine";
    std::string reid_path = "weights/unified/unified_reid_fp16.engine";
    std::string input_src = "0"; // Default camera 0 hoặc video path

    if (argc > 1) input_src = argv[1];
    if (argc > 2) yolo_path = argv[2];
    if (argc > 3) reid_path = argv[3];

    // Khởi tạo các module
    std::cout << "[1/4] Khởi tạo Camera Motion Compensation (CMC)..." << std::endl;
    SparseOpticalFlowCMC cmc(100, 0.01, 30.0, 0.35);

    std::cout << "[2/4] Khởi tạo YOLO Detector: " << yolo_path << "..." << std::endl;
    YoloDetector detector(yolo_path, 0.55f, 0.45f);

    std::cout << "[3/4] Khởi tạo Unified ReID Extractor: " << reid_path << "..." << std::endl;
    ReidExtractor reid(reid_path);

    std::cout << "[4/4] Khởi tạo UKF Tracker (cosine=0.42, iou=0.7)..." << std::endl;
    Tracker tracker(30, 3, 0.42, 0.7, 10.0);

    Visualizer visualizer;

    // Mở video stream hoặc camera
    cv::VideoCapture cap;
    if (input_src.size() == 1 && std::isdigit(input_src[0])) {
        cap.open(std::stoi(input_src));
    } else {
        cap.open(input_src);
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Không thể mở nguồn video/camera: " << input_src << std::endl;
        return -1;
    }

    std::cout << "[START] Bắt đầu vòng lặp xử lý video..." << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    double total_time_ms = 0.0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_idx++;

        auto t_start = std::chrono::high_resolution_clock::now();

        // Bước 1: CMC (Ước lượng chuyển động máy ảnh)
        cv::Mat H = cmc.apply(frame);

        // Bước 2: Nhận diện vật thể YOLO
        std::vector<Detection> dets = detector.detect(frame);

        // Bước 3: Trích xuất đặc trưng ReID
        reid.extract(frame, dets);

        // Bước 4: Cập nhật bộ lọc và liên kết quỹ đạo (UKF Tracker)
        tracker.update(dets, frame.cols, frame.rows, H, frame_idx);

        // Bước 5: Lấy kết quả và hiển thị
        auto results = tracker.get_results(true, frame_idx);

        auto t_end = std::chrono::high_resolution_clock::now();
        double frame_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        total_time_ms += frame_time_ms;
        double current_fps = 1000.0 / std::max(frame_time_ms, 1e-3);

        visualizer.draw(frame, results, current_fps);

        if (frame_idx % 30 == 0) {
            std::cout << "Frame " << frame_idx << " | Tracks: " << results.size() 
                      << " | Latency: " << frame_time_ms << " ms (" << current_fps << " FPS)" << std::endl;
        }

        // Thoát nếu bấm phím 'q'
        // cv::imshow("UKF-DeepSORT C++", frame);
        // if (cv::waitKey(1) == 'q') break;
    }

    double avg_fps = (frame_idx > 0) ? (frame_idx * 1000.0 / total_time_ms) : 0.0;
    std::cout << "========================================================" << std::endl;
    std::cout << "Hoàn thành! Tổng số frame: " << frame_idx 
              << " | FPS trung bình: " << avg_fps << std::endl;
    std::cout << "========================================================" << std::endl;

    return 0;
}
