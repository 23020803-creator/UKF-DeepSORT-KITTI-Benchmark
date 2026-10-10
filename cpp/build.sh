#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "    BẮT ĐẦU BIÊN DỊCH DỰ ÁN UKF-DEEPSORT C++ CHO JETSON   "
echo "=========================================================="

# 1. Kiểm tra RAM & Swap
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
TOTAL_SWAP=$(free -m | awk '/^Swap:/{print $2}')

echo "[INFO] Dung lượng RAM vật lý: ${TOTAL_MEM} MB"
echo "[INFO] Dung lượng Swap: ${TOTAL_SWAP} MB"

if [ "$TOTAL_SWAP" -lt 2000 ]; then
    echo "[CẢNH BÁO] Swap nhỏ hơn 2GB! Khuyên bạn nên tạo 4GB-6GB Swap để tránh tràn RAM khi biên dịch."
fi

# 2. Tạo thư mục build
mkdir -p build
cd build

# 3. Chạy CMake
echo "[INFO] Đang cấu hình CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. Biên dịch mã nguồn bằng 4 luồng
echo "[INFO] Đang biên dịch mã nguồn C++..."
make -j4

echo "=========================================================="
echo "    BIÊN DỊCH THÀNH CÔNG! File thực thi: build/ukf_deepsort"
echo "=========================================================="
echo "Cách chạy thử nghiệm:"
echo "  ./ukf_deepsort <video_path_hoặc_camera_id> <yolo_engine> <reid_engine>"
echo "Ví dụ:"
echo "  ./ukf_deepsort 0 ../weights/yolo26n_fp16.engine ../weights/unified/unified_reid_fp16.engine"
