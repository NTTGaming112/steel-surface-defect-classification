"""
Prediction logic for steel defect detection
"""
import numpy as np
from PIL import Image
from src.core import config
from src.core.models import model_manager
from src.core.preprocessing import extract_sift_features

def predict_defect(image):
    """Predict steel surface defect using YOLO + SVM"""
    if image is None:
        return "⚠️ Please upload an image", None, ""
    
    try:
        # Convert to numpy array
        image_np = np.array(image) if isinstance(image, Image.Image) else image
        
        # Step 1: YOLO Detection
        print("🔍 YOLO detecting defects...")
        yolo_results = model_manager.yolo_model(image_np, conf=config.YOLO_CONFIDENCE)
        
        has_defect = len(yolo_results[0].boxes) > 0
        yolo_image = yolo_results[0].plot() if has_defect else image_np.copy()
        
        if has_defect:
            print(f"✅ YOLO detected {len(yolo_results[0].boxes)} defect(s)")
        else:
            print("ℹ️ No clear defects detected by YOLO, proceeding with SVM classification")
        
        # Step 2: SVM Classification
        print("🤖 SVM classifying defect type...")
        sift_features = extract_sift_features(image_np)
        features_scaled = model_manager.scaler.transform(sift_features.reshape(1, -1))
        
        prediction = model_manager.svm_model.predict(features_scaled)[0]
        probabilities = model_manager.svm_model.predict_proba(features_scaled)[0]
        predicted_class_idx = list(model_manager.svm_model.classes_).index(prediction)
        confidence_percent = probabilities[predicted_class_idx] * 100
        
        # Top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "Loại lỗi": model_manager.svm_model.classes_[idx],
                "Xác suất": f"{probabilities[idx] * 100:.1f}%"
            }
            for idx in top_indices
        ]
        
        # Build output text
        output_text = "## 🎯 Detection Results\n\n"
        output_text += f"### YOLO Detection: **{'Defect Found' if has_defect else 'No Clear Defect'}**\n\n"
        if has_defect:
            output_text += f"### Detected Regions: **{len(yolo_results[0].boxes)}**\n\n"
        output_text += f"### SVM Classification: **{prediction}**\n\n"
        output_text += f"### Confidence: **{confidence_percent:.1f}%**\n\n"
        output_text += "---\n\n### 📊 Top 3 Predictions:\n\n"
        
        for i, pred in enumerate(top_predictions, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            output_text += f"{emoji} **{pred['Loại lỗi']}**: {pred['Xác suất']}\n\n"
        
        # JSON results
        json_results = {
            "YOLO Detection": "Defect Found" if has_defect else "No Clear Defect",
            "Số vùng lỗi": len(yolo_results[0].boxes) if has_defect else 0,
            "SVM Classification": prediction,
            "Độ tin cậy": f"{confidence_percent:.1f}%",
            "Top 3 Dự đoán": top_predictions
        }
        
        processed_pil = Image.fromarray(yolo_image)
        return json_results, processed_pil, output_text
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {traceback.format_exc()}")
        return f"❌ Error: {str(e)}", None, str(e)
