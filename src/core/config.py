"""
Configuration file for Steel Defect Detection AI
"""

# Class names for defect types
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 
               'Rolled-in_scale', 'Scratches']

# Model paths
YOLO_MODEL_PATH = "models/best.pt"
SIFT_EXTRACTOR_PATH = "models/sift_extractor_svm_param1.pkl"
SVM_MODEL_PATH = "models/best_svm_SIFT_param1.pkl"
SCALER_PATH = "models/scaler_svm_SIFT_param1.pkl"

# YOLO11 detection confidence threshold
YOLO_CONFIDENCE = 0.25

# Image preprocessing parameters
IMAGE_SIZE = (200, 200)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# UI configuration
APP_TITLE = "Steel Defect Detection AI"
APP_DESCRIPTION = "🔍 Steel Surface Defect Detection using Machine Learning"
MAX_WIDTH = 1400
