#include "models/TrtEngine.hpp"
#include <fstream>
#include <iostream>

#ifdef USE_TENSORRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;
#endif

namespace ukf_deepsort {

TrtEngine::TrtEngine(const std::string& engine_path) : engine_path_(engine_path) {
    init_engine();
}

TrtEngine::~TrtEngine() {
#ifdef USE_TENSORRT
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    for (void* buf : device_buffers_) {
        if (buf) cudaFree(buf);
    }
    device_buffers_.clear();

    if (context_) context_->destroy();
    if (engine_) engine_->destroy();
    if (runtime_) runtime_->destroy();
#endif
}

void TrtEngine::init_engine() {
#ifdef USE_TENSORRT
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[ERROR] Không thể mở file TensorRT Engine: " << engine_path_ << std::endl;
        is_valid_ = false;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "[ERROR] Lỗi deserialize TensorRT Engine từ file!" << std::endl;
        is_valid_ = false;
        return;
    }

    context_ = engine_->createExecutionContext();
    cudaStreamCreate(&stream_);

    int nb_bindings = engine_->getNbBindings();
    device_buffers_.resize(nb_bindings, nullptr);
    buffer_sizes_.resize(nb_bindings, 0);
    is_input_buffer_.resize(nb_bindings, false);

    for (int i = 0; i < nb_bindings; ++i) {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine_->getBindingDataType(i);

        size_t total_elements = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            total_elements *= dims.d[d];
        }

        size_t elem_size = 4; // FP32 mặc định
        if (dtype == nvinfer1::DataType::kHALF) elem_size = 2;
        else if (dtype == nvinfer1::DataType::kINT8) elem_size = 1;

        size_t byte_size = total_elements * elem_size;
        buffer_sizes_[i] = byte_size;
        is_input_buffer_[i] = engine_->bindingIsInput(i);

        cudaMalloc(&device_buffers_[i], byte_size);
    }

    is_valid_ = true;
    std::cout << "[SUCCESS] Nạp thành công TensorRT Engine: " << engine_path_ << std::endl;
#else
    std::cout << "[INFO] Build không kích hoạt cờ USE_TENSORRT. TrtEngine ở chế độ Dummy." << std::endl;
    is_valid_ = false;
#endif
}

bool TrtEngine::infer(const std::vector<void*>& host_inputs, const std::vector<void*>& host_outputs) {
#ifdef USE_TENSORRT
    if (!is_valid_) return false;

    // 1. Copy host to device
    size_t in_idx = 0;
    size_t out_idx = 0;
    int nb_bindings = engine_->getNbBindings();

    for (int i = 0; i < nb_bindings; ++i) {
        if (is_input_buffer_[i]) {
            if (in_idx < host_inputs.size() && host_inputs[in_idx] != nullptr) {
                cudaMemcpyAsync(device_buffers_[i], host_inputs[in_idx], buffer_sizes_[i], cudaMemcpyHostToDevice, stream_);
            }
            in_idx++;
        }
    }

    // 2. Chạy enqueueV2
    context_->enqueueV2(device_buffers_.data(), stream_, nullptr);

    // 3. Copy device to host
    for (int i = 0; i < nb_bindings; ++i) {
        if (!is_input_buffer_[i]) {
            if (out_idx < host_outputs.size() && host_outputs[out_idx] != nullptr) {
                cudaMemcpyAsync(host_outputs[out_idx], device_buffers_[i], buffer_sizes_[i], cudaMemcpyDeviceToHost, stream_);
            }
            out_idx++;
        }
    }

    cudaStreamSynchronize(stream_);
    return true;
#else
    (void)host_inputs;
    (void)host_outputs;
    return false;
#endif
}

size_t TrtEngine::get_input_size(int index) const {
#ifdef USE_TENSORRT
    int cur = 0;
    for (size_t i = 0; i < is_input_buffer_.size(); ++i) {
        if (is_input_buffer_[i]) {
            if (cur == index) return buffer_sizes_[i];
            cur++;
        }
    }
#endif
    (void)index;
    return 0;
}

size_t TrtEngine::get_output_size(int index) const {
#ifdef USE_TENSORRT
    int cur = 0;
    for (size_t i = 0; i < is_input_buffer_.size(); ++i) {
        if (!is_input_buffer_[i]) {
            if (cur == index) return buffer_sizes_[i];
            cur++;
        }
    }
#endif
    (void)index;
    return 0;
}

int TrtEngine::get_input_binding_idx(const std::string& name) const {
#ifdef USE_TENSORRT
    if (engine_) return engine_->getBindingIndex(name.c_str());
#endif
    (void)name;
    return -1;
}

int TrtEngine::get_output_binding_idx(const std::string& name) const {
#ifdef USE_TENSORRT
    if (engine_) return engine_->getBindingIndex(name.c_str());
#endif
    (void)name;
    return -1;
}

} // namespace ukf_deepsort
