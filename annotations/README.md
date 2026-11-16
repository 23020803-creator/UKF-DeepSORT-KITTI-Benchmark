# Dữ liệu gán nhãn và chuẩn bị huấn luyện

Thư mục này chứa các bộ ảnh/nhãn đang chuẩn bị hoặc kiểm duyệt trên CVAT.
Dữ liệu nguồn và bộ dữ liệu dùng để huấn luyện đặt trong `datasets/`;
kết quả chạy tracking/đánh giá đặt trong `outputs/`.

```text
annotations/
├── README.md
└── cvat/
    ├── exports/                              # ZIP nhãn của data1 … data10
    └── cvat_random150_conf010_20261027/       # Bộ 150 ảnh để sửa nhãn
        ├── README_CVAT.md                    # Hướng dẫn upload và kiểm duyệt
        ├── 150_images.zip
        ├── 150_annotations_yolo1.1.zip
        ├── selection.csv / selection.json    # Truy vết ảnh nguồn
        ├── summary.json                      # Thống kê và checksum ZIP
        ├── subset/                           # Ảnh và nhãn auto-label gốc
        ├── cvat_ready/                       # Nhãn đã chuẩn hóa cho CVAT
        └── ...                               # Preview, ảnh tham khảo, script của bộ mẫu
```

## Đóng gói nhãn

Chạy từ thư mục gốc dự án:

```bash
python tools/package_cvat_yolo.py --base datasets --start 1 --end 10
```

Mặc định ZIP được lưu tại `annotations/cvat/exports/`. Dùng `--out` để chọn
thư mục khác. Các ZIP này chứa nhãn; ảnh được cung cấp riêng khi tạo task CVAT.

## Bộ 150 ảnh

Bộ mẫu trước đây nằm tại `outputs/cvat_random150_conf010_20261027/`, nay ở
`annotations/cvat/cvat_random150_conf010_20261027/`. Các ZIP trước đây trong
`cvat_zips/` đã chuyển sang `annotations/cvat/exports/`.

Xem [README_CVAT.md](cvat/cvat_random150_conf010_20261027/README_CVAT.md) trong bộ
mẫu để upload ảnh và nhãn. Nhãn auto-label cần được kiểm duyệt trước khi dùng
huấn luyện. Giữ `selection.csv`, `selection.json` và `summary.json` cùng bộ mẫu
để truy vết nguồn và kiểm tra tính toàn vẹn.

`annotations/cvat/` được Git bỏ qua vì chứa dữ liệu cục bộ, ảnh và ZIP dung lượng
lớn. Hướng dẫn này và công cụ đóng gói được quản lý trong mã nguồn.
