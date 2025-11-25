# Steel Surface Defect Detection

Dự án phát hiện khuyết tật bề mặt thép sử dụng Computer Vision và Machine Learning với nhiều mô hình khác nhau.

## 📁 Cấu trúc dự án

```
project/
├── data/              # Dữ liệu thô (NEU-DET)
├── models/            # Các mô hình ML đã train (SVM, RF, KNN)
├── checkpoints/       # Trọng số của các mô hình
├── demo/              # Ứng dụng web Gradio
├── notebooks/         # Jupyter notebooks (EDA, thử nghiệm)
├── docs/              # Tài liệu
└── README.md
```

## 📋 Mô tả

Hệ thống phát hiện và phân loại 6 loại khuyết tật trên bề mặt thép:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

## 🎯 Dataset

**NEU Surface Defect Database**

- Training: 1440 ảnh (240 ảnh/class × 6 classes)
- Validation: 360 ảnh (60 ảnh/class × 6 classes)
- Tổng: 1800 ảnh
- Kích thước: 200x200 pixels (grayscale sau preprocessing)

## 🚀 Models

Chi tiết về quá trình huấn luyện và kết quả của từng mô hình có trong các notebooks tương ứng:

- **SVM (Support Vector Machine)**: `notebooks/support-vector-machine.ipynb`
- **Random Forest**: `notebooks/random-forest.ipynb`
- **KNN (K-Nearest Neighbors)**: `notebooks/k-nearest-neighbor.ipynb`

Kết quả dưới đây là ví dụ về hiệu suất tốt nhất đạt được trong các thử nghiệm.

## 🔧 Feature Extraction

### SIFT (Scale-Invariant Feature Transform)

- Sử dụng Bag of Visual Words với các kích thước vocabulary khác nhau (ví dụ: 100, 200).
- Dùng MiniBatchKMeans để tạo vocabulary.

### LBP (Local Binary Pattern)

- Thử nghiệm với các phương pháp LBP khác nhau (`default`, `uniform`).
- Trích xuất histogram từ ảnh LBP.

### Preprocessing

- Cải thiện độ tương phản với CLAHE (clipLimit=2.0, tileGridSize=8x8).
- Thay đổi kích thước ảnh về 200x200.
- Chuyển đổi sang ảnh xám.

## 📦 Cài đặt

```bash
pip install -r requirements.txt
```

## 🎮 Sử dụng

### 1. Training Models

Mở và chạy các notebooks trong thư mục `notebooks/` để huấn luyện lại các mô hình:

- `support-vector-machine.ipynb`
- `random-forest.ipynb`
- `k-nearest-neighbor.ipynb`

### 2. Web Demo

Để chạy ứng dụng demo, di chuyển vào thư mục `demo` và chạy file `app.py`:

```bash
cd demo
python app.py
```

Truy cập: http://127.0.0.1:7860

## 🎨 Web Interface Features

- 📤 Tải lên ảnh khuyết tật.
- 🔍 Hiển thị ảnh sau khi tiền xử lý.
- 🎯 3 dự đoán hàng đầu với điểm tin cậy.
- 📊 Bảng kết quả chi tiết.
- 🎨 Giao diện tùy chỉnh.

**Note**:

- Tất cả models sử dụng combined features (SIFT BoVW 100 + LBP 64 = 164 features)
- Training set: 1440 samples, Test set: 360 samples
- Hyperparameter optimization: Optuna với TPE Sampler

## 🔬 Hyperparameter Optimization

Sử dụng **Optuna** với:

- 1000 trials cho mỗi feature set
- 3-fold cross-validation
- TPE Sampler
- Automatic checkpoint saving

## 📝 Notes

- Tất cả models sử dụng StandardScaler
- SIFT extractor được save để inference

## 📄 License

MIT License
