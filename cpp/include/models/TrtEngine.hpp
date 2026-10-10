#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace ukf_deepsort {

/**
 * @brief Lớp bọc (Wrapper) quản lý nạp và suy luận TensorRT Engine trên GPU NVIDIA Jetson Nano.
 */
class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    bool is_valid() const { return is_valid_; }

    /**
     * @brief Thực thi suy luận bất đồng bộ trên luồng CUDA.
     * @param host_inputs Mảng con trỏ dữ liệu đầu vào trên RAM Host (CPU).
     * @param host_outputs Mảng con trỏ dữ liệu đầu ra trên RAM Host (CPU).
     */
    bool infer(const std::vector<void*>& host_inputs, const std::vector<void*>& host_outputs);

    // Kích thước buffer
    size_t get_input_size(int index = 0) const;
    size_t get_output_size(int index = 0) const;

    int get_input_binding_idx(const std::string& name) const;
    int get_output_binding_idx(const std::string& name) const;

private:
    std::string engine_path_;
    bool is_valid_{false};

#ifdef USE_TENSORRT
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    cudaStream_t stream_{nullptr};

    std::vector<void*> device_buffers_;
    std::vector<size_t> buffer_sizes_;
    std::vector<bool> is_input_buffer_;
#endif

    void init_engine();
};

} // namespace ukf_deepsort
