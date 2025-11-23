# Báo Cáo Đồ Án: Phân Loại Khuyết Tật Bề Mặt Thép

---

## 1. GIỚI THIỆU

### 1.1. Tên Đề Tài

**Phân loại khuyết tật bề mặt thép sử dụng Computer Vision và Machine Learning**

### 1.2. Lý Do Chọn Đề Tài

- **Ý nghĩa thực tiễn**: Trong công nghiệp sản xuất thép, việc phát hiện và phân loại khuyết tật bề mặt là công đoạn quan trọng để đảm bảo chất lượng sản phẩm. Việc kiểm tra thủ công tốn kém thời gian, nhân lực và dễ sai sót.

- **Ứng dụng AI/ML**: Áp dụng các thuật toán Computer Vision và Machine Learning giúp tự động hóa quy trình kiểm tra, tăng tốc độ, độ chính xác và giảm chi phí sản xuất.

- **Thách thức kỹ thuật**: Các khuyết tật trên bề mặt thép có đặc điểm phức tạp, đa dạng về hình dạng, kích thước và mức độ nghiêm trọng, đòi hỏi phải nghiên cứu các phương pháp trích xuất đặc trưng và mô hình phân loại phù hợp.

### 1.3. Phát Biểu Bài Toán

#### Input (Đầu vào)

- **Dữ liệu**: Ảnh RGB bề mặt thép kích thước 200×200 pixels
- **Nguồn**: NEU Surface Defect Database
- **Định dạng**: JPG/PNG/BMP
- **Đặc điểm**: Ảnh chụp bề mặt thép công nghiệp với các loại khuyết tật khác nhau

#### Output (Đầu ra)

- **Loại khuyết tật**: Phân loại ảnh đầu vào vào 1 trong 6 lớp khuyết tật:

  1. **Crazing** (Vết nứt mịn)
  2. **Inclusion** (Tạp chất)
  3. **Patches** (Vùng không đồng nhất)
  4. **Pitted Surface** (Bề mặt lõm)
  5. **Rolled-in Scale** (Vảy cán vào)
  6. **Scratches** (Vết trầy xước)

- **Độ tin cậy**: Xác suất/độ chắc chắn của dự đoán (nếu model hỗ trợ)

#### Mục tiêu

Xây dựng hệ thống phân loại tự động với độ chính xác cao (>90%) để hỗ trợ kiểm tra chất lượng trong dây chuyền sản xuất thép.

---

## 2. PHƯƠNG PHÁP

### 2.1. Tiền Xử Lý (Preprocessing)

#### 2.1.1. Tên Phương Pháp

- **Grayscale Conversion**: Chuyển đổi ảnh màu sang ảnh xám
- **Image Resizing**: Chuẩn hóa kích thước
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Tăng cường độ tương phản

#### 2.1.2. Ý Tưởng

- **Grayscale**: Giảm chiều dữ liệu từ 3 kênh màu (RGB) xuống 1 kênh, giảm độ phức tạp tính toán mà vẫn giữ được thông tin texture quan trọng cho phân loại khuyết tật.

- **Resizing**: Chuẩn hóa tất cả ảnh về kích thước 200×200 pixels để đảm bảo tính đồng nhất khi trích xuất đặc trưng.

- **CLAHE**: Áp dụng histogram equalization cục bộ trên từng vùng nhỏ của ảnh (tile 8×8) với giới hạn độ tương phản (clip limit = 2.0) để:
  - Tăng cường chi tiết trong vùng tối/sáng
  - Tránh khuếch đại nhiễu quá mức
  - Làm nổi bật đặc điểm khuyết tật

#### 2.1.3. Áp Dụng

**Pipeline tiền xử lý được áp dụng cho mọi ảnh trong dataset**:

1. **Đọc ảnh từ file**: Load ảnh màu RGB từ đường dẫn file

2. **Chuyển đổi sang grayscale**: Sử dụng cv2.cvtColor với color space BGR2GRAY để giảm từ 3 channels xuống 1 channel, giữ lại thông tin intensity

3. **Resize về kích thước chuẩn**: Tất cả ảnh được resize về 200×200 pixels bằng INTER_AREA interpolation (tốt cho downsampling)

4. **Tăng cường độ tương phản**: Áp dụng CLAHE với:
   - clipLimit = 2.0: Giới hạn khuếch đại để tránh nhiễu
   - tileGridSize = (8, 8): Chia ảnh thành lưới 8×8 tiles để equalization cục bộ

**Áp dụng cho**:

- Toàn bộ 1,200 ảnh training set
- Toàn bộ 600 ảnh validation set
- Được gọi trong các hàm extract_lbp() và SiftBowExtractor để đảm bảo tính nhất quán

---

### 2.2. Trích Xuất Đặc Trưng (Feature Extraction)

#### 2.2.1. Local Binary Pattern (LBP)

##### Tên Phương Pháp

**LBP (Local Binary Pattern)** - Mẫu nhị phân cục bộ

##### Ý Tưởng

LBP là descriptor texture đơn giản nhưng hiệu quả, mô tả kết cấu bề mặt bằng cách:

1. So sánh giá trị pixel trung tâm với 8 điểm lân cận xung quanh
2. Mã hóa kết quả thành chuỗi nhị phân 8-bit
3. Tạo histogram của các mẫu LBP làm vector đặc trưng

**Hai biến thể sử dụng**:

- **LBP Default**: Sử dụng tất cả 2^8 = 256 bins, giữ toàn bộ thông tin chi tiết
- **LBP Uniform**: Chỉ giữ các mẫu "uniform" (≤2 transitions), giảm xuống 11 bins, robust với nhiễu và rotation

##### Áp Dụng

**Trích xuất LBP features cho 2 cấu hình tham số**:

**Configuration 1 - LBP Default**:

- Radius = 1 pixel: So sánh với 8 neighbors trong vòng tròn bán kính 1
- n_points = 8: Lấy 8 điểm xung quanh pixel trung tâm
- Method = 'default': Giữ tất cả 256 patterns (2^8)
- Feature vector: 256 dimensions (histogram với 256 bins)

**Configuration 2 - LBP Uniform**:

- Radius = 1 pixel
- n_points = 8
- Method = 'uniform': Chỉ giữ uniform patterns (≤2 bit transitions)
- Feature vector: 11 dimensions (8 + 2 + 1 bins)

**Quy trình áp dụng**:

1. Tiền xử lý ảnh (grayscale + resize + CLAHE)
2. Tính LBP pattern cho mỗi pixel
3. Tạo histogram của các patterns
4. Normalize histogram thành probability distribution
5. Sử dụng histogram làm feature vector cho classification

**Kết quả**: Mỗi ảnh được biểu diễn bởi 1 vector 256-D (default) hoặc 11-D (uniform)

---

#### 2.2.2. SIFT + Bag-of-Words (BoW)

##### Tên Phương Pháp

**SIFT BoW (Scale-Invariant Feature Transform with Bag-of-Words)**

##### Ý Tưởng

- **SIFT**: Phát hiện và mô tả các keypoints trong ảnh bằng 128-dimensional descriptors, bất biến với scale, rotation và lighting changes.

- **Bag-of-Words**:
  1. Trích xuất SIFT descriptors từ tất cả ảnh training
  2. Clustering các descriptors thành K clusters (vocabulary) bằng K-Means
  3. Mỗi ảnh được biểu diễn bởi histogram tần suất xuất hiện của các visual words
  4. Histogram này trở thành vector đặc trưng cho classification

**Ưu điểm**:

- Bắt được local features quan trọng (cạnh, góc, texture patterns)
- Robust với biến đổi hình học
- Phù hợp với khuyết tật có đặc điểm đa dạng về shape và scale

##### Áp Dụng

**Quy trình trích xuất SIFT BoW gồm 2 phases**:

**Phase 1: Training - Xây dựng Visual Vocabulary**

1. Trích xuất SIFT descriptors từ TẤT CẢ 1,200 ảnh training:

   - Mỗi ảnh → ~100-500 keypoints
   - Mỗi keypoint → 1 descriptor 128-D
   - Tổng: ~150,000-600,000 descriptors

2. Clustering descriptors thành vocabulary:

   - Sử dụng MiniBatchKMeans (nhanh hơn KMeans thông thường)
   - Configuration 1: K = 100 clusters → 100 visual words
   - Configuration 2: K = 200 clusters → 200 visual words
   - Batch size = 200 để xử lý hiệu quả

3. Lưu vocabulary (cluster centers) để tái sử dụng

**Phase 2: Feature Extraction - Chuyển ảnh thành BoW histogram**

1. Với mỗi ảnh (train hoặc test):

   - Trích xuất SIFT descriptors
   - Gán mỗi descriptor vào visual word gần nhất (dùng vocabulary đã học)
   - Đếm tần suất xuất hiện của mỗi visual word
   - Normalize thành histogram probability

2. Feature vector cuối cùng:
   - Configuration 1: 100-D vector
   - Configuration 2: 200-D vector

**Lưu ý quan trọng**: Vocabulary CHỈ được xây dựng từ training set để tránh data leakage

---

### 2.3. Chuẩn Hóa Đặc Trưng

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- Sử dụng StandardScaler để chuẩn hóa về mean=0, std=1
- Fit trên training set, transform cho cả train và test

---

### 2.4. Mô Hình Phân Loại (Classification Models)

#### 2.4.1. Support Vector Machine (SVM)

##### Tên Phương Pháp

**SVM (Support Vector Machine)** với kernel RBF/Linear/Polynomial

##### Ý Tưởng

SVM tìm siêu phẳng tối ưu phân tách các classes trong không gian đặc trưng bằng cách:

- Maximize margin giữa các classes
- Sử dụng kernel trick để xử lý dữ liệu non-linearly separable
- Tối ưu hóa với regularization parameter C

**Kernel functions**:

- **RBF (Radial Basis Function)**: Phù hợp cho dữ liệu phức tạp, non-linear
- **Linear**: Đơn giản, nhanh, phù hợp với dữ liệu linearly separable
- **Polynomial**: Bắt được polynomial relationships

##### Áp Dụng

**Training pipeline cho SVM với 4 feature sets**:

**Bước 1: Hyperparameter Optimization với Optuna**

- Cho mỗi feature set (LBP_default, LBP_uniform, SIFT_100, SIFT_200):

  - Định nghĩa search space:

    - C: [0.1, 1000] với log scale (regularization parameter)
    - Kernel: thử 3 loại ['rbf', 'linear', 'poly']
    - Gamma: ['scale', 'auto'] cho RBF và polynomial
    - Degree: [2, 5] cho polynomial kernel

  - Chạy 100 trials với TPESampler (Bayesian optimization)
  - Mỗi trial: Train model với hyperparameters được suggest → Đánh giá bằng 3-fold CV
  - Chọn configuration có CV accuracy cao nhất

**Bước 2: Train Final Model**

- Dùng best hyperparameters từ Optuna
- Train trên toàn bộ training set (1,200 samples)
- Predict trên validation set (600 samples)
- Tính test accuracy và classification metrics

**Bước 3: Save Artifacts**

- Lưu trained model (best*svm*{feature_name}.pkl)
- Lưu scaler (scaler*svm*{feature_name}.pkl)
- Lưu Optuna study để phân tích (study*svm*{feature_name}.pkl)
- Lưu metadata: best params, accuracies, timestamp

**Kết quả**: 4 SVM models được tối ưu hóa cho 4 feature sets khác nhau

---

#### 2.4.2. K-Nearest Neighbors (KNN)

##### Tên Phương Pháp

**KNN (K-Nearest Neighbors)** - Láng giềng gần nhất

##### Ý Tưởng

KNN là thuật toán instance-based learning:

- Không cần training phase
- Dự đoán class của sample mới dựa trên K neighbors gần nhất
- Sử dụng voting (uniform) hoặc weighted voting (distance-based)
- Distance metric: Euclidean, Manhattan, Minkowski

**Ưu điểm**:

- Đơn giản, dễ implement
- Không giả định về phân phối dữ liệu
- Hiệu quả với dữ liệu có decision boundary phức tạp

##### Áp Dụng

**Training pipeline cho KNN với 4 feature sets**:

**Bước 1: Hyperparameter Optimization**

- Cho mỗi feature set:

  - Search space:

    - n_neighbors: [3, 15] - số lượng neighbors gần nhất
    - weights:
      - 'uniform': tất cả neighbors có trọng số bằng nhau
      - 'distance': neighbors gần hơn có trọng số cao hơn (1/distance)
    - metric: thử 3 distance metrics
      - 'euclidean': L2 distance
      - 'manhattan': L1 distance
      - 'minkowski': generalized distance

  - 100 trials với Optuna TPESampler
  - Đánh giá bằng 3-fold cross-validation
  - Chọn best configuration

**Bước 2: Train & Evaluate**

- KNN không có training phase thực sự (lazy learning)
- Chỉ cần lưu trữ training data
- Prediction: Tìm K neighbors gần nhất và vote
- Sử dụng n_jobs=-1 để parallel processing

**Bước 3: Save Models**

- Lưu fitted KNN model với best hyperparameters
- Lưu scaler đã fit trên training data
- Lưu metadata và Optuna study

**Đặc điểm**: KNN yêu cầu StandardScaler vì nhạy cảm với scale của features

---

#### 2.4.3. Random Forest (RF)

##### Tên Phương Pháp

**Random Forest** - Rừng ngẫu nhiên

##### Ý Tưởng

Random Forest là ensemble learning method:

- Xây dựng multiple decision trees trên random subsets của data
- Mỗi tree voting cho class prediction
- Final prediction: majority voting
- Giảm overfitting thông qua bagging và feature randomness

**Ưu điểm**:

- Robust với noise và outliers
- Xử lý tốt high-dimensional data
- Cung cấp feature importance
- Ít nhạy cảm với hyperparameters

##### Áp Dụng

**Training pipeline cho Random Forest với 4 feature sets**:

**Bước 1: Hyperparameter Optimization**

- Cho mỗi feature set:

  - Search space rộng với 5 hyperparameters:

    - **n_estimators** [50, 300]: Số lượng decision trees trong forest
      - Nhiều trees → accuracy cao hơn nhưng chậm hơn
    - **max_depth** [5, 30]: Độ sâu tối đa của mỗi tree
      - Sâu → overfitting, nông → underfitting
    - **min_samples_split** [2, 10]: Số samples tối thiểu để split node
      - Cao → regularization mạnh hơn
    - **min_samples_leaf** [1, 5]: Số samples tối thiểu tại leaf node
      - Cao → smoother decision boundaries
    - **max_features** ['sqrt', 'log2']: Số features xét khi split
      - 'sqrt': √n_features (tăng diversity)
      - 'log2': log₂(n_features) (regularization mạnh hơn)

  - 100 trials với Optuna
  - 3-fold CV để đánh giá

**Bước 2: Train Ensemble**

- Build forest với best hyperparameters
- Mỗi tree train trên random bootstrap sample
- Mỗi split xét random subset của features
- Sử dụng n_jobs=-1 để parallel training
- Random state = 42 để reproducibility

**Bước 3: Prediction & Evaluation**

- Mỗi tree vote cho 1 class
- Final prediction: majority voting
- Test trên validation set

**Ưu điểm**: Random Forest ít cần feature scaling, robust với outliers

---

### 2.5. Tối Ưu Hóa Hyperparameters

#### Phương pháp: Optuna

- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Optimization**: Maximize cross-validation accuracy
- **Cross-validation**: 3-fold CV
- **Number of trials**: 100 cho mỗi model + feature combination

#### Workflow

1. Định nghĩa search space cho hyperparameters
2. Optuna tự động sample và đánh giá các configurations
3. Sử dụng Bayesian optimization để hội tụ nhanh
4. Chọn configuration tốt nhất dựa trên CV accuracy

---

## 3. THỰC NGHIỆM

### 3.1. Dataset

#### 3.1.1. Thông Tin Dataset

- **Tên**: NEU Surface Defect Database
- **Nguồn**: Northeastern University, China
- **Số lượng ảnh**: 1,800 ảnh grayscale
- **Kích thước**: 200×200 pixels
- **Số lượng classes**: 6 loại khuyết tật
- **Phân phối**: 300 ảnh/class (balanced)

#### 3.1.2. Các Lớp Khuyết Tật

| Class | Tên Tiếng Anh   | Mô Tả                     | Số Lượng Ảnh |
| ----- | --------------- | ------------------------- | ------------ |
| 1     | Crazing         | Vết nứt mịn dạng lưới     | 300          |
| 2     | Inclusion       | Tạp chất không thuần nhất | 300          |
| 3     | Patches         | Vùng không đồng đều       | 300          |
| 4     | Pitted Surface  | Bề mặt có lỗ nhỏ          | 300          |
| 5     | Rolled-in Scale | Vảy oxit bám vào          | 300          |
| 6     | Scratches       | Vết trầy xước             | 300          |

#### 3.1.3. Phân Chia Dataset

```
Training Set: 1,200 ảnh (200 ảnh/class)
Validation Set: 600 ảnh (100 ảnh/class)
Tỉ lệ: 67% train / 33% validation
```

---

### 3.2. Độ Đo Đánh Giá

#### 3.2.1. Metrics Sử Dụng

1. **Accuracy**: Tỉ lệ dự đoán đúng trên tổng số samples

   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: Tỉ lệ dự đoán đúng trong các positive predictions

   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall**: Tỉ lệ phát hiện được trong các positive samples thực tế

   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Harmonic mean của Precision và Recall

   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **Cross-Validation Accuracy**: Accuracy trung bình trên 3-fold CV (training phase)

---

### 3.3. Cấu Hình Thực Nghiệm

#### 3.3.1. Môi Trường

- **Hardware**: [Điền cấu hình máy của bạn]

  - CPU: [...]
  - RAM: [...]
  - GPU: [...] (nếu có)

- **Software**:
  - Python: 3.x
  - OpenCV: 4.x
  - scikit-learn: 1.x
  - scikit-image: 0.x
  - Optuna: 3.x

#### 3.3.2. Cấu Hình Feature Extraction

| Feature Type | Parameter  | Value                |
| ------------ | ---------- | -------------------- |
| **LBP**      |
|              | radius     | 1                    |
|              | n_points   | 8                    |
|              | method     | 'default' (256 bins) |
|              | method     | 'uniform' (11 bins)  |
| **SIFT BoW** |
|              | vocab_size | 100                  |
|              | vocab_size | 200                  |
|              | batch_size | 200                  |

#### 3.3.3. Tổ Hợp Thực Nghiệm

**Tổng số models**: 3 algorithms × 4 feature sets = **12 models**

| Model | Feature Extraction | Feature Params     | Feature Dims |
| ----- | ------------------ | ------------------ | ------------ |
| SVM-1 | LBP                | default (256 bins) | 256          |
| SVM-2 | LBP                | uniform (11 bins)  | 11           |
| SVM-3 | SIFT BoW           | vocab_size=100     | 100          |
| SVM-4 | SIFT BoW           | vocab_size=200     | 200          |
| KNN-1 | LBP                | default (256 bins) | 256          |
| KNN-2 | LBP                | uniform (11 bins)  | 11           |
| KNN-3 | SIFT BoW           | vocab_size=100     | 100          |
| KNN-4 | SIFT BoW           | vocab_size=200     | 200          |
| RF-1  | LBP                | default (256 bins) | 256          |
| RF-2  | LBP                | uniform (11 bins)  | 11           |
| RF-3  | SIFT BoW           | vocab_size=100     | 100          |
| RF-4  | SIFT BoW           | vocab_size=200     | 200          |

---

### 3.4. Kết Quả

#### 3.4.1. Bảng Kết Quả Tổng Hợp

**[Điền kết quả sau khi chạy thực nghiệm]**

| Model | Feature     | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score |
| ----- | ----------- | ----------- | ------------- | --------- | ------ | -------- |
| SVM-1 | LBP_default |             |               |           |        |          |
| SVM-2 | LBP_uniform |             |               |           |        |          |
| SVM-3 | SIFT_100    |             |               |           |        |          |
| SVM-4 | SIFT_200    |             |               |           |        |          |
| KNN-1 | LBP_default |             |               |           |        |          |
| KNN-2 | LBP_uniform |             |               |           |        |          |
| KNN-3 | SIFT_100    |             |               |           |        |          |
| KNN-4 | SIFT_200    |             |               |           |        |          |
| RF-1  | LBP_default |             |               |           |        |          |
| RF-2  | LBP_uniform |             |               |           |        |          |
| RF-3  | SIFT_100    |             |               |           |        |          |
| RF-4  | SIFT_200    |             |               |           |        |          |

#### 3.4.2. Best Hyperparameters

**[Điền best params từ Optuna sau khi chạy]**

##### SVM Best Configurations

```python
# Ví dụ:
SVM-1 (LBP_default):
  - kernel: 'rbf'
  - C: 10.5
  - gamma: 'scale'

SVM-2 (LBP_uniform):
  - kernel: 'linear'
  - C: 50.2

# ... tiếp tục cho các models khác
```

##### KNN Best Configurations

```python
# Điền sau khi chạy
```

##### RF Best Configurations

```python
# Điền sau khi chạy
```

#### 3.4.3. Confusion Matrix - Model Tốt Nhất

**[Thêm confusion matrix visualization sau khi chạy]**

```
Ví dụ format:
              Predicted
           Cr  In  Pa  Pi  Ro  Sc
Actual Cr [95   2   1   0   1   1]
       In [ 1  92   3   2   2   0]
       Pa [ 0   1  94   3   1   1]
       Pi [ 1   2   2  93   1   1]
       Ro [ 2   1   0   1  95   1]
       Sc [ 1   0   2   1   0  96]
```

#### 3.4.4. Classification Report - Model Tốt Nhất

**[Điền classification report chi tiết]**

```
              precision    recall  f1-score   support

     Crazing       0.95      0.95      0.95       100
   Inclusion       0.92      0.92      0.92       100
     Patches       0.94      0.94      0.94       100
Pitted_surface     0.93      0.93      0.93       100
Rolled-in_scale    0.95      0.95      0.95       100
   Scratches       0.96      0.96      0.96       100

    accuracy                           0.94       600
   macro avg       0.94      0.94      0.94       600
weighted avg       0.94      0.94      0.94       600
```

---

### 3.5. Minh Họa Kết Quả Model Tốt Nhất

#### 3.5.1. Dự Đoán Đúng (3 ví dụ)

**[Thêm 3 ảnh với predictions đúng]**

```
Image 1:
- True Label: Crazing
- Predicted: Crazing
- Confidence: 0.98
[Hình ảnh]

Image 2:
- True Label: Scratches
- Predicted: Scratches
- Confidence: 0.95
[Hình ảnh]

Image 3:
- True Label: Inclusion
- Predicted: Inclusion
- Confidence: 0.92
[Hình ảnh]
```

#### 3.5.2. Dự Đoán Sai (3 ví dụ)

**[Thêm 3 ảnh với predictions sai và phân tích]**

```
Image 1:
- True Label: Patches
- Predicted: Pitted_surface
- Confidence: 0.65
- Phân tích: Vùng khuyết tật có texture tương tự
[Hình ảnh]

Image 2:
- True Label: Rolled-in_scale
- Predicted: Inclusion
- Confidence: 0.58
- Phân tích: Cả hai đều có vùng tối không đều
[Hình ảnh]

Image 3:
- True Label: Crazing
- Predicted: Scratches
- Confidence: 0.71
- Phân tích: Pattern có đường nét tương tự nhau
[Hình ảnh]
```

---

### 3.6. Phân Tích Kết Quả

#### 3.6.1. So Sánh Các Feature Extraction Methods

**[Điền phân tích dựa trên kết quả]**

1. **LBP vs SIFT**:

   - LBP: [...]
   - SIFT: [...]
   - Nhận xét: [...]

2. **LBP Default vs Uniform**:

   - Default (256 bins): [...]
   - Uniform (11 bins): [...]
   - Nhận xét: [...]

3. **SIFT vocab_size 100 vs 200**:
   - Vocab 100: [...]
   - Vocab 200: [...]
   - Nhận xét: [...]

#### 3.6.2. So Sánh Các Models

**[Điền phân tích]**

1. **SVM**:

   - Điểm mạnh: [...]
   - Điểm yếu: [...]
   - Phù hợp với: [...]

2. **KNN**:

   - Điểm mạnh: [...]
   - Điểm yếu: [...]
   - Phù hợp với: [...]

3. **Random Forest**:
   - Điểm mạnh: [...]
   - Điểm yếu: [...]
   - Phù hợp với: [...]

#### 3.6.3. Khuyết Tật Khó Phân Loại

**[Điền dựa trên confusion matrix]**

1. Cặp classes dễ nhầm lẫn nhất: [...]
2. Nguyên nhân: [...]
3. Đề xuất cải thiện: [...]

---

## 4. KẾT LUẬN

### 4.1. Tổng Kết

**[Điền sau khi có đầy đủ kết quả]**

- Model tốt nhất: [Tên model] với Test Accuracy: [X.XX%]
- Feature extraction tốt nhất: [LBP/SIFT] với configuration: [...]
- Hyperparameters tối ưu: [...]

### 4.2. Đóng Góp Của Đề Tài

1. **Nghiên cứu và so sánh** nhiều phương pháp feature extraction và classification algorithms
2. **Áp dụng thành công** các kỹ thuật Computer Vision và Machine Learning vào bài toán thực tế
3. **Đạt độ chính xác cao** [X%] trong phân loại khuyết tật bề mặt thép
4. **Tự động hóa quy trình** kiểm tra chất lượng, giảm thời gian và chi phí

### 4.3. Hạn Chế

1. **Dataset**: Còn hạn chế về số lượng và đa dạng điều kiện chụp
2. **Feature engineering**: Chỉ thử nghiệm LBP và SIFT, chưa khảo sát deep features
3. **Computational cost**: SIFT BoW tốn thời gian trích xuất và clustering
4. **Generalization**: Chưa test trên dữ liệu từ nhà máy thực tế

### 4.4. Hướng Phát Triển

1. **Deep Learning**:

   - Thử nghiệm CNN architectures (ResNet, EfficientNet, Vision Transformer)
   - Transfer learning từ pretrained models
   - End-to-end learning thay vì hand-crafted features

2. **Data Augmentation**:

   - Tăng cường dữ liệu bằng rotation, flip, noise injection
   - Synthetic data generation với GANs

3. **Real-time Detection**:

   - Tối ưu hóa inference speed
   - Deploy model lên edge devices
   - Integration với dây chuyền sản xuất

4. **Multi-task Learning**:
   - Vừa phân loại vừa localization khuyết tật
   - Severity assessment

---

## PHỤ LỤC

### A. Cấu Trúc Thư Mục

```
project/
├── data/
│   └── NEU-DET/
│       ├── train/
│       │   ├── images/
│       │   └── annotations/
│       └── validation/
│           ├── images/
│           └── annotations/
├── checkpoints/
│   ├── study_svm_LBP_param1.pkl
│   ├── study_svm_SIFT_param1.pkl
│   └── ...
├── models/
│   ├── best_svm_LBP_param1.pkl
│   ├── scaler_svm_LBP_param1.pkl
│   ├── metadata_svm_LBP_param1.json
│   └── ...
├── cv-project-svm.ipynb
├── cv-project-knn.ipynb
├── cv-project-randomforest.ipynb
└── README.md
```

### B. Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
tqdm>=4.65.0
optuna>=3.3.0
joblib>=1.3.0
```

### C. Hướng Dẫn Chạy Code

1. **Cài đặt dependencies**:

```bash
pip install -r requirements.txt
```

2. **Download dataset**:

```python
import kagglehub
path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
```

3. **Chạy notebooks**:

- `cv-project-svm.ipynb`: Training SVM models
- `cv-project-knn.ipynb`: Training KNN models
- `cv-project-randomforest.ipynb`: Training Random Forest models

4. **Load trained models**:

```python
import joblib
model = joblib.load('models/best_svm_LBP_param1.pkl')
scaler = joblib.load('models/scaler_svm_LBP_param1.pkl')
```

---

## TÀI LIỆU THAM KHẢO

1. **Dataset**:

   - He, Y., Song, K., Meng, Q., & Yan, Y. (2019). "An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features". IEEE Transactions on Instrumentation and Measurement.

2. **LBP**:

   - Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns". IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971-987.

3. **SIFT**:

   - Lowe, D. G. (2004). "Distinctive image features from scale-invariant keypoints". International Journal of Computer Vision, 60(2), 91-110.

4. **SVM**:

   - Cortes, C., & Vapnik, V. (1995). "Support-vector networks". Machine Learning, 20(3), 273-297.

5. **Random Forest**:

   - Breiman, L. (2001). "Random forests". Machine Learning, 45(1), 5-32.

6. **Optuna**:
   - Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework". KDD.

---

**Ngày hoàn thành**: [Điền ngày]

**Sinh viên thực hiện**: [Tên của bạn]

**Giảng viên hướng dẫn**: [Tên giảng viên]
