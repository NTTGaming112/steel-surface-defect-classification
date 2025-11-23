"""
Model loading and initialization
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import joblib
import cv2
from ultralytics import YOLO
from src.core import config

class ModelManager:
    """Manages loading and access to all models"""
    
    def __init__(self):
        self.yolo_model = None
        self.sift_extractor = None
        self.svm_model = None
        self.scaler = None
        self.clahe = None
        
    def load_models(self):
        """Load all models and preprocessors"""
        import sys
        import utils
        
        # Make utils available for pickle to find SiftBowExtractor
        sys.modules['__main__'].SiftBowExtractor = utils.SiftBowExtractor
        
        print("🔄 Loading models...")
        
        # Load YOLO model
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
        print("✅ YOLO model loaded")
        
        # Load SIFT extractor
        self.sift_extractor = joblib.load(config.SIFT_EXTRACTOR_PATH)
        print("✅ SIFT extractor loaded")
        
        # Load SVM model
        self.svm_model = joblib.load(config.SVM_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        print("✅ SVM model loaded")
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE
        )
        
        return self

# Global model manager instance
model_manager = ModelManager()
