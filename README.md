# Steel Defect Detection AI

Hệ thống phát hiện và phân loại lỗi bề mặt thép sử dụng YOLO11 + SVM

## 📁 Cấu trúc thư mục

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
└── checkpoints/               # For saving results (optional)
```

## 🚀 Cách chạy

```bash
python app.py
```

## 🔧 Pipeline

1. **YOLO11 Detection** - Phát hiện vùng lỗi
2. **SVM Classification** - Phân loại loại lỗi (6 classes)
3. **SIFT Features** - Trích xuất đặc trưng

## 📊 Classes

- Crazing
- Inclusion
- Patches
- Pitted_surface
- Rolled-in_scale
- Scratches
