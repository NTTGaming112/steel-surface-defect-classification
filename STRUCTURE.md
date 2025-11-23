# Cấu trúc thư mục dự án

```
project/
├── data/                          # Dữ liệu gốc
│   └── NEU-DET/
│       ├── train/
│       │   ├── images/
│       │   └── annotations/
│       └── validation/
│           ├── images/
│           └── annotations/
│
├── datasets/                      # Datasets đã xử lý
│   └── yolo_dataset/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── data.yaml
│
├── models/                        # Models đã train (traditional ML)
│   ├── metadata_knn_*.json
│   ├── metadata_svm_*.json
│   └── metadata_rf_*.json
│
├── checkpoints/                   # YOLO checkpoints
│   └── yolo11n/
│       └── weights/
│           └── best.pt
│
├── demo/                          # Ứng dụng Gradio
│   ├── app.py
│   ├── utils.py
│   ├── src/
│   │   ├── core/
│   │   │   ├── models.py
│   │   │   └── prediction.py
│   │   └── ui/
│   │       └── ui_components.py
│   ├── models/                    # Symlink hoặc copy từ ../models
│   └── checkpoints/               # Symlink hoặc copy từ ../checkpoints
│
├── notebooks/                     # Jupyter notebooks
│   ├── eda.ipynb
│   ├── cv-project-knn.ipynb
│   ├── cv-project-svm.ipynb
│   └── cv-project-randomforest.ipynb
│
├── scripts/                       # Scripts tiện ích
│   ├── convert_neudet_to_yolo.py
│   └── train_yolo.py
│
├── docs/                          # Tài liệu
│   ├── REPORT.md
│   ├── BaoCaoCuoiKiCV.docx
│   └── CV_report.docx
│
├── results/                       # Kết quả, hình ảnh
│   ├── knn.jpg
│   ├── svm.jpg
│   └── rf.jpg
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Mô tả các thư mục chính:

- **data/**: Dữ liệu thô, không được commit vào git
- **datasets/**: Dữ liệu đã xử lý sẵn sàng cho training
- **models/**: Models ML truyền thống (KNN, SVM, Random Forest)
- **checkpoints/**: Weights của YOLO models
- **demo/**: Ứng dụng web Gradio độc lập
- **notebooks/**: Jupyter notebooks cho EDA và experiments
- **scripts/**: Scripts Python cho data processing và training
- **docs/**: Tài liệu dự án
- **results/**: Visualization và kết quả experiments
