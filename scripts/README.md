# Hướng dẫn sử dụng Scripts

## 1. Convert Dataset (convert_neudet_to_yolo.py)

Chuyển đổi NEU-DET dataset từ format XML sang YOLO format.

```bash
cd scripts
python convert_neudet_to_yolo.py
```

**Output:**

- `datasets/yolo_dataset/images/train/` - 1151 ảnh train
- `datasets/yolo_dataset/images/val/` - 360 ảnh validation
- `datasets/yolo_dataset/images/test/` - 288 ảnh test
- `datasets/yolo_dataset/labels/` - Labels tương ứng
- `datasets/yolo_dataset/data.yaml` - Config file cho YOLO

## 2. Train YOLO Model (train_yolo.py)

Train YOLO11n model để detect steel defects.

```bash
cd scripts
python train_yolo.py
```

**Cấu hình:**

- Model: YOLO11n (pretrained)
- Epochs: 50
- Image size: 640
- Batch size: 16

**Output:**

- Checkpoints: `checkpoints/steel_defect_yolo11n/weights/`
  - `best.pt` - Best model
  - `last.pt` - Last epoch model
- Training logs và metrics

## 3. Demo Application

Chạy ứng dụng Gradio để test models:

```bash
cd demo
python app.py
```

Ứng dụng sẽ chạy tại: http://localhost:7860

## Cấu trúc Dữ liệu

```
data/NEU-DET/          # Raw data (không commit)
datasets/              # Processed data
  └── yolo_dataset/    # YOLO format
models/                # Traditional ML models
checkpoints/           # YOLO weights
```

## Notebooks

Các Jupyter notebooks trong `notebooks/`:

- `eda.ipynb` - Exploratory Data Analysis
- `cv-project-knn.ipynb` - KNN classifier
- `cv-project-svm.ipynb` - SVM classifier
- `cv-project-randomforest.ipynb` - Random Forest classifier
