"""
Utility functions for NEU Surface Defect Detection
Extracted from cv-project.ipynb for reuse in app.py
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# CLAHE preprocessor
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(image, use_clahe=True):
    """Đọc ảnh, chuyển xám, resize 200x200 và áp dụng CLAHE."""
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh từ: {image}")
    
    # Convert từ PIL/numpy nếu cần
    if hasattr(image, 'convert'):  # PIL Image
        image = np.array(image)
    
    # Chuyển sang grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        raise ValueError("Định dạng ảnh không được hỗ trợ.")
    
    # Resize về 200x200
    if gray.shape != (200, 200):
        gray = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_AREA)
    
    # Áp dụng CLAHE
    if use_clahe:
        gray = clahe.apply(gray)
    
    return gray


def extract_lbp(image):
    """Trích xuất đặc trưng LBP từ ảnh."""
    gray = preprocess_image(image)
    
    n_points = 8
    radius = 1
    lbp = local_binary_pattern(gray, n_points, radius, method='default')
    
    # Tính histogram
    (hist, _) = np.histogram(lbp.ravel(), 
                             bins=np.arange(0, n_points**2 + 1), 
                             range=(0, n_points**2))
    
    # Chuẩn hóa histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    return hist


class SiftBowExtractor:
    """SIFT Bag-of-Visual-Words Extractor"""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.kmeans = None
        self.vocabulary = None
    
    def _get_sift_descriptors(self, image):
        """Trích xuất SIFT descriptors từ ảnh."""
        sift = cv2.SIFT_create()
        gray = preprocess_image(image)
        _, descriptors = sift.detectAndCompute(gray, None)
        return descriptors
    
    def transform_single(self, image):
        """Transform một ảnh thành SIFT BoVW histogram."""
        descriptors = self._get_sift_descriptors(image)
        hist = np.zeros(self.vocab_size, dtype=float)
        
        if descriptors is not None and self.kmeans is not None:
            visual_words = self.kmeans.predict(descriptors)
            hist, _ = np.histogram(visual_words, bins=np.arange(self.vocab_size + 1))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-6)
        
        return hist
