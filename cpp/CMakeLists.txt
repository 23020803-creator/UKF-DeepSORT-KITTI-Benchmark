cmake_minimum_required(VERSION 3.10)
project(UKF_DeepSORT_Jetson LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Cờ tối ưu hóa kiến trúc ARMv8-A cho Jetson Nano (Cortex-A57)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "[JETSON] Phát hiện nền tảng ARM aarch64! Kích hoạt tối ưu Cortex-A57.")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8-a -mcpu=cortex-a57 -funroll-loops -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -DNDEBUG")
endif()

# 1. Thư viện OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Tìm thấy OpenCV: " ${OpenCV_VERSION} " tại " ${OpenCV_INCLUDE_DIRS})

# 2. Thư viện Eigen 3 (Header-only)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Tìm thấy Eigen3 tại: " ${EIGEN3_INCLUDE_DIR})

# 3. Kiểm tra CUDA & TensorRT
option(ENABLE_TENSORRT "Kích hoạt NVIDIA TensorRT" ON)

if(ENABLE_TENSORRT)
    find_package(CUDA)
    find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
    find_path(NVINFER_INCLUDE_DIR NvInfer.h HINTS /usr/include/aarch64-linux-gnu /usr/local/cuda/include)

    if(CUDA_FOUND AND NVINFER_LIB AND NVINFER_INCLUDE_DIR)
        message(STATUS "Tìm thấy TensorRT tại: " ${NVINFER_LIB})
        add_definitions(-DUSE_TENSORRT)
        set(HAS_TRT TRUE)
    else()
        message(WARNING "Không tìm thấy TensorRT! Sẽ build ở chế độ CPU/Dummy cho các module AI.")
        set(HAS_TRT FALSE)
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if(HAS_TRT)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${NVINFER_INCLUDE_DIR}
    )
endif()

file(GLOB_RECURSE SRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(ukf_deepsort 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp 
    ${SRC_FILES}
)

target_link_libraries(ukf_deepsort
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

if(HAS_TRT)
    target_link_libraries(ukf_deepsort
        PRIVATE
        ${CUDA_LIBRARIES}
        ${NVINFER_LIB}
    )
endif()
