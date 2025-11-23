# Steel Defect Detection AI - Demo

Hệ thống phát hiện và phân loại lỗi bề mặt thép sử dụng YOLO11 + SVM

## 📁 Cấu trúc

```
demo/
├── app.py                      # Main application
├── models/                     # Trained models & checkpoints
│   ├── best.pt                # YOLO detection model
│   ├── best_svm_SIFT_param1.pkl
│   ├── sift_extractor_svm_param1.pkl
│   └── scaler_svm_SIFT_param1.pkl
├── src/
│   ├── core/                  # Core logic
│   │   ├── config.py         # Configuration
│   │   ├── models.py         # Model management
│   │   ├── preprocessing.py  # Image preprocessing
│   │   ├── prediction.py     # Prediction logic
│   │   └── utils.py          # SIFT extractor
│   └── ui/                    # User interface
│       └── ui_components.py  # UI components & styling
├── checkpoints/               # For saving results (optional)
└── utils.py                   # Utility functions
```

## 🚀 Cài đặt và Chạy

### 1. Clone repository

```bash
git clone -b demo-only https://github.com/NTTGaming112/steel-surface-defect-classification.git
cd steel-surface-defect-classification
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install gradio opencv-python scikit-learn scikit-image joblib pillow numpy ultralytics
```

### 3. Chạy ứng dụng

```bash
cd demo
python app.py
```

Truy cập: http://127.0.0.1:7860

## 🔧 Pipeline

1. **YOLO11 Detection** - Phát hiện vùng lỗi trên bề mặt thép
2. **SVM Classification** - Phân loại loại lỗi (6 classes)
3. **SIFT Features** - Trích xuất đặc trưng từ vùng lỗi

## 📊 Classes

Hệ thống phát hiện 6 loại khuyết tật:

- **Crazing** - Vết nứt nhỏ
- **Inclusion** - Tạp chất
- **Patches** - Vết loang
- **Pitted Surface** - Bề mặt bị rỗ
- **Rolled-in Scale** - Vảy cuộn
- **Scratches** - Vết xước

## 🎨 Tính năng

- 📤 Upload ảnh defect
- 🔍 Hiển thị ảnh sau tiền xử lý
- 🎯 Top-3 predictions với confidence scores
- 📊 Detailed results table
- 🎨 Custom gradient theme

## 📈 Hiệu suất

- **YOLO11**: Object detection
- **SVM**: 98.33% test accuracy
- **Features**: SIFT + LBP (164 features)

## 📝 Lưu ý

- Models đã được train sẵn, không cần train lại
- Chỉ cần cài dependencies và chạy
- Hỗ trợ ảnh grayscale 200x200

## 🔗 Links

- **Full Project**: [Main Branch](https://github.com/NTTGaming112/steel-surface-defect-classification)
- **Dataset**: [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

## 📄 License

MIT License
