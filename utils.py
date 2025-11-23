"""
SIFT Bag-of-Visual-Words Extractor for Steel Defect Detection
"""

import cv2
import numpy as np

class SiftBowExtractor:
    """SIFT Bag-of-Visual-Words Extractor"""
    
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.kmeans = None
        self.vocabulary = None
    
    def _get_sift_descriptors(self, image):
        """Trích xuất SIFT descriptors từ ảnh."""
        sift = cv2.SIFT_create()
        # Image should already be preprocessed (grayscale, resized)
        _, descriptors = sift.detectAndCompute(image, None)
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
