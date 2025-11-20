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
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 2rem !important;
}

/* --- HEADER MODERN --- */
.header-container {
    text-align: center;
    padding: 3rem 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    border: none;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
}

.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    text-transform: uppercase;
}

.header-subtitle {
    font-size: 1.2rem;
    color: #64748b;
    font-weight: 500;
}

/* --- MAIN CONTENT CARDS --- */
.input-card, .output-card {
    background: var(--background-fill-primary);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    height: 100%;
    border: 1px solid var(--border-color-primary);
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--body-text-color);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title:before {
    content: '';
    width: 4px;
    height: 24px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 2px;
}

/* --- RESULT BOX MODERN --- */
.result-card {
    background: var(--background-fill-secondary);
    border: 2px solid var(--border-color-primary);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.15);
}

.result-title-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
}

.result-defect-name {
    font-size: 3rem;
    font-weight: 800;
    margin: 15px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}

.confidence-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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

/* --- BUTTONS MODERN --- */
#predict-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    border-radius: 15px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

#predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
}

#predict-btn:active {
    transform: translateY(0);
}

button[variant="secondary"] {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
    border: 2px solid #cbd5e1 !important;
    color: #475569 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

button[variant="secondary"]:hover {
    background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%) !important;
    border-color: #94a3b8 !important;
}

/* --- IMAGE INPUT STYLING --- */
#img_input {
    border-radius: 20px !important;
    overflow: hidden !important;
    border: 3px dashed #cbd5e1 !important;
}

#img_input:hover {
    border-color: #667eea !important;
}

/* --- TABS STYLING --- */
.tabs {
    background: transparent !important;
}

.tab-nav button {
    font-weight: 600 !important;
    color: #64748b !important;
    border-radius: 12px 12px 0 0 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
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
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="color: var(--body-text-color-subdued); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; 
                        font-weight: 700; margin-bottom: 15px;">Kết quả phân tích</div>
            <div style="font-size: 3rem; font-weight: 800; margin: 15px 0; color: var(--body-text-color); letter-spacing: -1px;">
                {defect_type}
            </div>
            
            <style>
                .dark .dynamic-badge {{ color: {dark_badge_color} !important; border-color: {dark_badge_color} !important; }}
            </style>
            <div class="confidence-badge dynamic-badge" style="{badge_style}">
                Độ tin cậy: {confidence}
            </div>
        </div>
        
        <div style="border-top: 2px solid var(--border-color-primary); padding-top: 20px;">
            <h4 style="margin-bottom: 20px; color: var(--body-text-color); font-weight: 700; font-size: 1.1rem;">📈 Chi tiết Top 3:</h4>
    """
    
    for i, pred in enumerate(top3):
        width = float(pred['Xác suất'].strip('%'))
        # Màu thanh bar gradient
        if i == 0:
            bar_color = "#667eea"  # Top 1: Tím
        elif i == 1:
            bar_color = "#764ba2"  # Top 2: Tím đậm
        else:
            bar_color = "#94a3b8"  # Top 3: Xám
        
        html_content += f"""
        <div class="prediction-row">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items: center;">
                <span style="font-weight: 700; color: var(--body-text-color); font-size: 1rem;">{pred['Loại lỗi']}</span>
                <span style="font-weight: 700; color: #667eea; font-size: 1.1rem;">{pred['Xác suất']}</span>
            </div>
            <div style="background: var(--border-color-primary); height: 12px; border-radius: 6px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {bar_color}, {bar_color}); width: {width}%; height: 100%; border-radius: 6px; transition: width 0.5s;"></div>
            </div>
        </div>
        """
        
    html_content += "</div></div>"
    
    return html_content, proc_img, json_res

# ============================================================================
# 4. GIAO DIỆN GRADIO
# ============================================================================

# Theme hiện đại với màu gradient tím
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.slate,
    font=("Poppins", "sans-serif")
)

with gr.Blocks(title="Steel Defect AI", theme=theme, css=custom_css) as demo:
    
    # --- Header Modern ---
    with gr.Row(elem_classes="header-container"):
        with gr.Column():
            gr.HTML("""
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 15px; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🏭
                </div>
                <div class="header-title">Steel Defect Detection AI</div>
                <div class="header-subtitle">🔍 Phát hiện khuyết tật bề mặt thép bằng Machine Learning</div>
                <div style="margin-top: 15px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                    <span style="background: rgba(102, 126, 234, 0.1); padding: 8px 20px; border-radius: 20px; 
                                 color: #667eea; font-weight: 600; font-size: 0.9rem;">⚡ SVM Model</span>
                    <span style="background: rgba(118, 75, 162, 0.1); padding: 8px 20px; border-radius: 20px; 
                                 color: #764ba2; font-weight: 600; font-size: 0.9rem;">🎯 SIFT + LBP Features</span>
                    <span style="background: rgba(16, 185, 129, 0.1); padding: 8px 20px; border-radius: 20px; 
                                 color: #10b981; font-weight: 600; font-size: 0.9rem;">📊 NEU-DET Dataset</span>
                </div>
            </div>
            """)

    # --- Main Content ---
    with gr.Row():
        # Cột Trái: Input
        with gr.Column(scale=5, elem_classes="input-card"):
            gr.HTML('<div class="section-title">📤 Tải ảnh lên</div>')
            
            img_input = gr.Image(
                type="pil", 
                label="", 
                height=420,
                sources=["upload", "clipboard"],
                elem_id="img_input"
            )
            
            gr.HTML("""
            <div style="margin: 15px 0; padding: 15px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        border-left: 4px solid #f59e0b; border-radius: 12px;">
                <div style="font-weight: 600; color: #92400e; margin-bottom: 5px;">💡 Hướng dẫn:</div>
                <div style="color: #78350f; font-size: 0.9rem;">
                    • Upload ảnh khuyết tật thép<br>
                    • Kích thước đề xuất: 200x200px
                </div>
            </div>
            """)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa ảnh", variant="secondary", size="lg")
                predict_btn = gr.Button("🚀 Phân tích ngay", variant="primary", size="lg", elem_id="predict-btn")

        # Cột Phải: Output
        with gr.Column(scale=6, elem_classes="output-card"):
            gr.HTML('<div class="section-title">📊 Kết quả phân tích</div>')
            
            with gr.Tabs():
                with gr.TabItem("🎯 Dashboard"):
                    result_html = gr.HTML(
                        label="",
                        value="""
                        <div style='text-align: center; padding: 80px 40px; 
                                    background: var(--background-fill-secondary);
                                    border: 3px dashed var(--border-color-primary); border-radius: 20px;'>
                            <div style='font-size: 4rem; margin-bottom: 20px; opacity: 0.3;'>⏳</div>
                            <div style='font-size: 1.8rem; font-weight: 700; color: var(--body-text-color); margin-bottom: 10px;'>
                                Chờ phân tích...
                            </div>
                            <div style='color: var(--body-text-color-subdued); font-size: 1rem;'>
                                Vui lòng tải ảnh lên và nhấn <strong>Phân tích ngay</strong>
                            </div>
                        </div>
                        """
                    )
                
                with gr.TabItem("🖼️ Tiền xử lý"):
                    processed_output = gr.Image(label="", interactive=False, height=400)
                
                with gr.TabItem("📋 Raw Data"):
                    output_json = gr.JSON(label="")

    # --- Info Cards ---
    gr.HTML("""
    <div style="margin-top: 2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08); border: 1px solid var(--border-color-primary);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🎯</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">98.33% Accuracy</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">Độ chính xác test set</div>
        </div>
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08); border: 1px solid var(--border-color-primary);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">⚡</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">SVM Linear</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">Support Vector Machine</div>
        </div>
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08); border: 1px solid var(--border-color-primary);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🔍</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">SIFT + LBP</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">164 combined features</div>
        </div>
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08); border: 1px solid var(--border-color-primary);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">📊</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">6 Classes</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">NEU-DET Dataset</div>
        </div>
    </div>
    """)
    
    # --- Footer ---
    gr.HTML("""
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; 
                background: var(--background-fill-primary); border-radius: 20px; backdrop-filter: blur(10px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1); border: 1px solid var(--border-color-primary);">
        <div style="color: var(--body-text-color); font-size: 1rem; margin-bottom: 8px; font-weight: 700;">
            Steel Surface Defect Detection AI
        </div>
        <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">
            © 2025 NEU-DET Project | Trained on NEU Surface Defect Database
        </div>
    </div>
    """)

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
            """<div style='text-align: center; padding: 80px 40px; 
                        background: var(--background-fill-secondary);
                        border: 3px dashed var(--border-color-primary); border-radius: 20px;'>
                    <div style='font-size: 4rem; margin-bottom: 20px; opacity: 0.5;'>✨</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: #10b981; margin-bottom: 10px;'>
                        Đã làm mới!
                    </div>
                    <div style='color: var(--body-text-color-subdued); font-size: 1rem;'>
                        Sẵn sàng phân tích ảnh mới
                    </div>
               </div>""", 
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