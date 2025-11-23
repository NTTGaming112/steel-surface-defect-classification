"""
Image preprocessing and feature extraction functions
"""
import numpy as np
import cv2
from PIL import Image
from src.core import config
from src.core.models import model_manager

def preprocess_image(img):
    """Preprocess image: grayscale, resize, CLAHE"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # Resize to standard size
    resized = cv2.resize(gray, config.IMAGE_SIZE)
    
    # Apply CLAHE enhancement
    enhanced = model_manager.clahe.apply(resized)
    
    return enhanced

def extract_sift_features(img):
    """Extract SIFT features from image"""
    preprocessed = preprocess_image(img)
    features = model_manager.sift_extractor.transform_single(preprocessed)
    return features
