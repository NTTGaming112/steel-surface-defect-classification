# Steel Surface Defect Detection

Dự án phát hiện khuyết tật bề mặt thép sử dụng Computer Vision và Machine Learning với nhiều mô hình khác nhau.

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

### 1. SVM (Support Vector Machine)

- **Test Accuracy**: 98.33%
- **CV Accuracy**: 97.43%
- **Features**: SIFT + LBP (164 features)
- **Best Params**: kernel=linear, C=0.122
- **File**: `cv-project.ipynb`

### 2. Decision Tree

- **Test Accuracy**: 96.39%
- **CV Accuracy**: 95.49%
- **Features**: SIFT + LBP (164 features)
- **Best Params**: max_depth=28, criterion=entropy, min_samples_split=15
- **File**: `cv-project-decisiontree.ipynb`

### 3. KNN (K-Nearest Neighbors)

- **Test Accuracy**: 96.67%
- **CV Accuracy**: 95.97%
- **Features**: SIFT + LBP (164 features)
- **Best Params**: n_neighbors=6, metric=minkowski (p=1), algorithm=kd_tree
- **File**: `cv-project-knn.ipynb`

## 🔧 Feature Extraction

### SIFT (Scale-Invariant Feature Transform)

- Bag of Visual Words với vocab_size=100
- MiniBatchKMeans clustering

### LBP (Local Binary Pattern)

- 8 points, radius 1
- 64-bin histogram

### Preprocessing

- CLAHE enhancement (clipLimit=2.0, tileGridSize=8x8)
- Resize to 200x200
- Grayscale conversion

## 📦 Cài đặt

```bash
pip install opencv-python scikit-learn scikit-image gradio joblib pillow numpy pandas matplotlib seaborn tqdm optuna
```

## 🎮 Sử dụng

### 1. Training Models

Chạy các notebook để train models:

```bash
jupyter notebook cv-project.ipynb          # SVM
jupyter notebook cv-project-decisiontree.ipynb  # Decision Tree
jupyter notebook cv-project-knn.ipynb      # KNN
```

### 2. Web Demo

Chạy Gradio web interface:

```bash
python app.py
```

Truy cập: http://127.0.0.1:7860

## 📁 Cấu trúc thư mục

```
project/
├── app.py                              # Gradio web interface
├── utils.py                            # Shared utility functions
├── cv-project.ipynb                    # SVM notebook
├── cv-project-decisiontree.ipynb       # Decision Tree notebook
├── cv-project-knn.ipynb                # KNN notebook
├── eda.ipynb                           # Exploratory Data Analysis
├── models/                             # Trained models (*.pkl not included in git)
│   ├── best_svm_ALL.pkl               # Best SVM model (SIFT+LBP)
│   ├── best_svm_LBP.pkl               # SVM model (LBP only)
│   ├── best_svm_SIFT.pkl              # SVM model (SIFT only)
│   ├── best_dt_ALL.pkl                # Best Decision Tree model
│   ├── best_knn_ALL.pkl               # Best KNN model
│   ├── sift_extractor.pkl             # SIFT BoVW extractor
│   ├── scaler_sift_lbp.pkl            # StandardScaler for combined features
│   └── metadata_*.json                # Model metadata files
├── checkpoints/                        # Optuna studies (*.pkl not included in git)
│   ├── study_ALL.pkl                  # SVM optimization study
│   ├── study_dt_ALL.pkl               # Decision Tree study
│   └── study_knn_ALL.pkl              # KNN study
├── demo/                               # Demo images (optional)
└── archive/                            # Dataset (not included in git)
    └── NEU-DET/
        ├── train/
        │   ├── images/                # Training images
        │   └── annotations/           # XML annotations
        └── validation/
            ├── images/                # Validation images
            └── annotations/           # XML annotations
```

## 🎨 Web Interface Features

- 📤 Upload ảnh defect
- 🔍 Hiển thị ảnh sau tiền xử lý
- 🎯 Top-3 predictions với confidence scores
- 📊 Detailed results table
- 🎨 Custom gradient theme

## 📊 Performance Comparison

| Model         | CV Accuracy | Test Accuracy | N Trials | Best Params                     |
| ------------- | ----------- | ------------- | -------- | ------------------------------- |
| SVM           | 97.43%      | **98.33%**    | 100      | kernel=linear, C=0.122          |
| KNN           | 95.97%      | 96.67%        | 100      | k=6, metric=minkowski (p=1)     |
| Decision Tree | 95.49%      | 96.39%        | 100      | max_depth=28, criterion=entropy |

**Note**:

- Tất cả models sử dụng combined features (SIFT BoVW 100 + LBP 64 = 164 features)
- Training set: 1440 samples, Test set: 360 samples
- Hyperparameter optimization: Optuna với TPE Sampler

## 🔬 Hyperparameter Optimization

Sử dụng **Optuna** với:

- 100 trials cho mỗi feature set
- 3-fold cross-validation
- TPE Sampler
- Automatic checkpoint saving

## 📝 Notes

- Models được train với scikit-learn 1.7.2
- Runtime có thể có version warning (1.6.1)
- Tất cả models sử dụng StandardScaler
- SIFT extractor được save để inference

## 👨‍💻 Author

Computer Vision Project - Steel Surface Defect Detection

## 📄 License

MIT License
