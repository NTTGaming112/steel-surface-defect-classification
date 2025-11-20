import gradio as gr
import numpy as np
import joblib
import os
from PIL import Image

# Import các hàm từ utils
from utils import preprocess_image, extract_lbp, SiftBowExtractor

# ============================================================================
# 1. CẤU HÌNH
# ============================================================================
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_surface', 
               'Rolled-in_scale', 'Scratches']
MODEL_DIR = "models"

# ============================================================================
# 2. LOAD MODEL VÀ DEPENDENCIES
# ============================================================================
print("🔄 Đang load model và dependencies...")

# Load SIFT extractor
sift_extractor = joblib.load(os.path.join(MODEL_DIR, "sift_extractor_svm.pkl"))
print("✅ Đã load SIFT extractor")

# Load model ALL (SIFT + LBP)
model = joblib.load(os.path.join(MODEL_DIR, "best_svm_ALL.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_svm_sift_lbp.pkl"))
print("✅ Đã load model ALL (98.3% accuracy)")

# ============================================================================
# 3. HÀM DỰ ĐOÁN
# ============================================================================
def predict_defect(image):
    """Dự đoán loại lỗi bề mặt thép từ ảnh."""
    if image is None:
        return "⚠️ Vui lòng upload ảnh", None, ""
    
    try:
        # Convert PIL to numpy nếu cần
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Trích xuất features
        lbp_features = extract_lbp(image)
        sift_features = sift_extractor.transform_single(image)
        
        # Kết hợp features (SIFT + LBP)
        combined_features = np.hstack([sift_features, lbp_features]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(combined_features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Lấy probability cho tất cả classes
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_class_idx = list(model.classes_).index(prediction)
        confidence_percent = probabilities[predicted_class_idx] * 100
        
        # Tạo top-k predictions
        top_k = 3  # Hiển thị top 3
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = []
        
        for idx in top_indices:
            class_name = model.classes_[idx]
            prob = probabilities[idx] * 100
            top_predictions.append({
                "Loại lỗi": class_name,
                "Xác suất": f"{prob:.1f}%"
            })
        
        # Tạo output
        output_text = f"## 🎯 Kết quả phát hiện lỗi\n\n"
        output_text += f"### Loại lỗi: **{prediction}**\n\n"
        output_text += f"### Độ tin cậy: **{confidence_percent:.1f}%**\n\n"
        output_text += f"---\n\n"
        output_text += f"### 📊 Top 3 dự đoán:\n\n"
        
        for i, pred in enumerate(top_predictions, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            output_text += f"{emoji} **{pred['Loại lỗi']}**: {pred['Xác suất']}\n\n"
        
        json_results = {
            "Loại lỗi": prediction,
            "Độ tin cậy": f"{confidence_percent:.1f}%",
            "Top 3 Dự đoán": top_predictions
        }
        
        # Ảnh sau tiền xử lý
        processed_img = preprocess_image(image)
        processed_pil = Image.fromarray(processed_img)
        
        return json_results, processed_pil, output_text
        
    except Exception as e:
        return f"❌ Lỗi: {str(e)}", None, str(e)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* --- CẤU HÌNH CHUNG --- */
body {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
}

/* --- HEADER (Thích ứng Sáng/Tối) --- */
.header-container {
    text-align: center;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    /* Sử dụng biến màu của Gradio để tự thích ứng */
    background: var(--background-fill-secondary); 
    border: 1px solid var(--border-color-primary);
    box-shadow: var(--shadow-drop);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    font-size: 1.1rem;
    color: var(--body-text-color-subdued);
}

/* --- INFO CARDS --- */
.info-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 1.5rem;
    transition: transform 0.2s;
}

.info-card:hover {
    transform: translateY(-3px);
    border-color: var(--primary-500);
}

.card-icon {
    font-size: 2rem;
    margin-bottom: 0.8rem;
}

.card-title {
    font-weight: 700;
    color: var(--body-text-color);
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.card-desc {
    color: var(--body-text-color-subdued);
    font-size: 0.95rem;
}

/* --- RESULT BOX STYLING --- */
/* Class này dùng cho HTML Output từ Python */
.result-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: var(--shadow-drop);
}

.result-title-label {
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-defect-name {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 10px 0;
    /* Màu mặc định cho Light mode */
    color: #1e40af; 
}

/* Override màu cho Dark mode */
.dark .result-defect-name {
    color: #60a5fa;
}

.confidence-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1rem;
}

.bar-container {
    background: var(--background-fill-primary);
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    margin-top: 5px;
}

.prediction-row {
    margin-bottom: 15px;
}

.pred-label {
    font-weight: 600;
    color: var(--body-text-color);
}

.pred-score {
    color: var(--body-text-color-subdued);
}

/* --- BUTTON --- */
#predict-btn {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    border: none;
    color: white;
    transition: all 0.3s;
}
#predict-btn:hover {
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
}
"""

# ============================================================================
# 3. HÀM WRAPPER ĐỂ RENDER HTML (Hỗ trợ Dark Mode)
# ============================================================================
def predict_wrapper(image):
    json_res, proc_img, text_res = predict_defect(image)
    
    if json_res is None:
        return (
            f"<div style='color: var(--error-text-color); padding: 20px; background: var(--background-fill-secondary); border-radius: 8px; border: 1px solid var(--error-border-color);'>{text_res}</div>",
            None,
            None
        )

    defect_type = json_res["Loại lỗi"]
    confidence = json_res["Độ tin cậy"]
    top3 = json_res["Top 3 Dự đoán"]
    
    # Xử lý màu sắc Badge dựa trên độ tin cậy
    conf_val = float(confidence.strip('%'))
    
    # Sử dụng biến CSS hoặc màu Hex có độ tương phản tốt trên cả 2 nền
    if conf_val > 85:
        badge_style = "background: rgba(34, 197, 94, 0.2); color: #16a34a; border: 1px solid #16a34a;" 
        # Dark mode override cho màu xanh lá sáng hơn
        dark_badge_color = "#4ade80" 
    elif conf_val > 50:
        badge_style = "background: rgba(234, 179, 8, 0.2); color: #ca8a04; border: 1px solid #ca8a04;"
        dark_badge_color = "#facc15"
    else:
        badge_style = "background: rgba(220, 38, 38, 0.2); color: #dc2626; border: 1px solid #dc2626;"
        dark_badge_color = "#f87171"

    # HTML Structure sử dụng các class đã định nghĩa trong CSS
    html_content = f"""
    <div class="result-card">
        <div style="text-align: center; margin-bottom: 25px;">
            <div class="result-title-label">Kết quả phân tích</div>
            <div class="result-defect-name">{defect_type}</div>
            
            <style>
                .dark .dynamic-badge {{ color: {dark_badge_color} !important; border-color: {dark_badge_color} !important; }}
            </style>
            <div class="confidence-badge dynamic-badge" style="{badge_style}">
                Độ tin cậy: {confidence}
            </div>
        </div>
        
        <div style="border-top: 1px solid var(--border-color-primary); padding-top: 20px;">
            <h4 style="margin-bottom: 15px; color: var(--body-text-color-subdued);">Chi tiết Top 3:</h4>
    """
    
    for i, pred in enumerate(top3):
        width = float(pred['Xác suất'].strip('%'))
        # Màu thanh bar: Top 1 màu xanh chủ đạo, còn lại màu xám
        bar_color = "var(--primary-500)" if i == 0 else "var(--neutral-400)"
        
        html_content += f"""
        <div class="prediction-row">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span class="pred-label">{pred['Loại lỗi']}</span>
                <span class="pred-score">{pred['Xác suất']}</span>
            </div>
            <div class="bar-container">
                <div style="background: {bar_color}; width: {width}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """
        
    html_content += "</div></div>"
    
    return html_content, proc_img, json_res

# ============================================================================
# 4. GIAO DIỆN GRADIO
# ============================================================================

# Sử dụng theme Soft có sẵn hỗ trợ tốt Dark Mode
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
).set(
    body_background_fill="var(--background-fill-primary)",
    block_background_fill="var(--background-fill-secondary)"
)

with gr.Blocks(title="Steel Defect AI", theme=theme, css=custom_css) as demo:
    
    # --- Header ---
    with gr.Row(elem_classes="header-container"):
        with gr.Column():
            gr.HTML("""
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 10px;">🏭</div>
                <div class="header-title">AI NHẬN DIỆN LỖI THÉP</div>
                <div class="header-subtitle">Phát hiện lỗi bề mặt thép cán nóng</div>
            </div>
            """)

    gr.Markdown("---")

    # --- Main Content ---
    with gr.Row():
        # Cột Trái: Input
        with gr.Column(scale=4):
            gr.Markdown("### 1. Hình ảnh đầu vào")
            with gr.Group():
                img_input = gr.Image(
                    type="pil", 
                    label="Tải ảnh lên", 
                    height=380,
                    sources=["upload", "clipboard"],
                    elem_id="img_input"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
                    predict_btn = gr.Button("🚀 PHÂN TÍCH NGAY", variant="primary", size="lg", elem_id="predict-btn")

        # Cột Phải: Output
        with gr.Column(scale=5):
            gr.Markdown("### 2. Kết quả phân tích")
            
            with gr.Tabs():
                with gr.TabItem("📊 Dashboard"):
                    result_html = gr.HTML(
                        label="Kết quả",
                        value="""
                        <div style='text-align: center; padding: 60px; color: var(--body-text-color-subdued); 
                                    border: 2px dashed var(--border-color-primary); border-radius: 12px;'>
                            <div style='font-size: 2rem; margin-bottom: 10px;'>Waiting...</div>
                            <div>Vui lòng tải ảnh lên và nhấn nút Phân Tích</div>
                        </div>
                        """
                    )
                
                with gr.TabItem("🖼️ Xử lý ảnh"):
                    processed_output = gr.Image(label="Ảnh sau khi qua bộ lọc", interactive=False)
                
                with gr.TabItem("📝 JSON"):
                    output_json = gr.JSON(label="Raw Data")

    # --- Footer ---
    gr.Markdown("---")
    gr.Markdown(
        """
        <div style="text-align: center; color: var(--body-text-color-subdued); opacity: 0.8;">
            © 2024 NEU Project. All rights reserved.<br>
            Model trained on NEU-DET Dataset.
        </div>
        """
    )

    # --- Events ---
    predict_btn.click(
        fn=predict_wrapper,
        inputs=img_input,
        outputs=[result_html, processed_output, output_json]
    )
    
    # Nút clear reset về trạng thái ban đầu
    clear_btn.click(
        lambda: (
            None, 
            """<div style='text-align: center; padding: 60px; color: var(--body-text-color-subdued); 
               border: 2px dashed var(--border-color-primary); border-radius: 12px;'>
               <div style='font-size: 2rem; margin-bottom: 10px;'>Ready</div>
               <div>Đã làm mới dữ liệu</div></div>""", 
            None, 
            None
        ),
        inputs=None,
        outputs=[img_input, result_html, processed_output, output_json]
    )

# ============================================================================
# 5. CHẠY APP
# ============================================================================
if __name__ == "__main__":
    print("🚀 Starting Adaptive UI Demo...")
    demo.launch(share=False)