"""
UI styling and CSS
"""

CUSTOM_CSS = """
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

.header-container {
    text-align: center;
    padding: 3rem 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    box-shadow: 0 20px 60px rgba(0,0,0,0.15);
}

.result-card {
    background: var(--background-fill-secondary);
    border: 2px solid var(--border-color-primary);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.15);
}

.confidence-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.prediction-row {
    margin-bottom: 15px;
}

#predict-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    border-radius: 15px !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

#predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
}
"""

def get_header_html():
    """Generate header HTML"""
    return """
    <div style="text-align: center;">
        <div style="font-size: 4rem; margin-bottom: 15px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏭
        </div>
        <div style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.8rem; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                    letter-spacing: -1px; text-transform: uppercase;">
            Steel Defect Detection AI
        </div>
        <div style="font-size: 1.2rem; color: #64748b; font-weight: 500;">
            🔍 Steel Surface Defect Detection using Machine Learning
        </div>
        <div style="margin-top: 15px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
            <span style="background: rgba(59, 130, 246, 0.1); padding: 8px 20px; border-radius: 20px; 
                         color: #3b82f6; font-weight: 600; font-size: 0.9rem;">🔍 YOLO Detection</span>
            <span style="background: rgba(168, 85, 247, 0.1); padding: 8px 20px; border-radius: 20px; 
                         color: #a855f7; font-weight: 600; font-size: 0.9rem;">🤖 SVM Classification</span>
            <span style="background: rgba(102, 126, 234, 0.1); padding: 8px 20px; border-radius: 20px; 
                         color: #667eea; font-weight: 600; font-size: 0.9rem;">🎯 SIFT Features</span>
            <span style="background: rgba(16, 185, 129, 0.1); padding: 8px 20px; border-radius: 20px; 
                         color: #10b981; font-weight: 600; font-size: 0.9rem;">📊 NEU-DET Dataset</span>
        </div>
    </div>
    """

def get_info_html():
    """Generate info box HTML"""
    return """
    <div style="margin: 15px 0; padding: 15px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                border-left: 4px solid #f59e0b; border-radius: 12px;">
        <div style="font-weight: 600; color: #92400e; margin-bottom: 5px;">💡 Analysis Pipeline:</div>
        <div style="color: #78350f; font-size: 0.9rem;">
            1️⃣ YOLO11 detects defect regions<br>
            2️⃣ SVM classifies defect type<br>
            • Recommended size: 200x200px
        </div>
    </div>
    """

def get_waiting_html():
    """Generate waiting state HTML"""
    return """
    <div style='text-align: center; padding: 80px 40px; 
                background: var(--background-fill-secondary);
                border: 3px dashed var(--border-color-primary); border-radius: 20px;'>
        <div style='font-size: 4rem; margin-bottom: 20px; opacity: 0.3;'>⏳</div>
        <div style='font-size: 1.8rem; font-weight: 700; color: var(--body-text-color); margin-bottom: 10px;'>
            Waiting for analysis...
        </div>
        <div style='color: var(--body-text-color-subdued); font-size: 1rem;'>
            Please upload an image and click <strong>Analyze</strong>
        </div>
    </div>
    """

def get_cleared_html():
    """Generate cleared state HTML"""
    return """
    <div style='text-align: center; padding: 80px 40px; 
                background: var(--background-fill-secondary);
                border: 3px dashed var(--border-color-primary); border-radius: 20px;'>
        <div style='font-size: 4rem; margin-bottom: 20px; opacity: 0.5;'>✨</div>
        <div style='font-size: 1.8rem; font-weight: 700; color: #10b981; margin-bottom: 10px;'>
            Cleared!
        </div>
        <div style='color: var(--body-text-color-subdued); font-size: 1rem;'>
            Ready for new analysis
        </div>
    </div>
    """

def get_stats_html():
    """Generate statistics cards HTML"""
    return """
    <div style="margin-top: 2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🔍</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">YOLO11</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">Object Detection</div>
        </div>
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🤖</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">SVM Classifier</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">Defect Classification</div>
        </div>
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🎯</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">SIFT Features</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">Feature Extraction</div>
        </div>
        <div style="background: var(--background-fill-primary); padding: 25px; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">📊</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: var(--body-text-color); margin-bottom: 5px;">6 Classes</div>
            <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">NEU-DET Dataset</div>
        </div>
    </div>
    """

def get_footer_html():
    """Generate footer HTML"""
    return """
    <div style="margin-top: 3rem; padding: 2rem; text-align: center; 
                background: var(--background-fill-primary); border-radius: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
        <div style="color: var(--body-text-color); font-size: 1rem; margin-bottom: 8px; font-weight: 700;">
            Steel Surface Defect Detection AI
        </div>
        <div style="color: var(--body-text-color-subdued); font-size: 0.9rem;">
            © 2025 NEU-DET Project | Trained on NEU Surface Defect Database
        </div>
    </div>
    """

def generate_result_html(json_res):
    """Generate HTML for prediction results"""
    yolo_status = json_res.get("YOLO Detection", "N/A")
    num_defects = json_res.get("Số vùng lỗi", 0)
    defect_type = json_res["SVM Classification"]
    confidence = json_res["Độ tin cậy"]
    top3 = json_res["Top 3 Dự đoán"]
    
    conf_val = float(confidence.strip('%'))
    
    # Determine badge color based on confidence
    if conf_val > 85:
        badge_style = "background: rgba(34, 197, 94, 0.2); color: #16a34a; border: 1px solid #16a34a;"
        dark_badge_color = "#4ade80"
    elif conf_val > 50:
        badge_style = "background: rgba(234, 179, 8, 0.2); color: #ca8a04; border: 1px solid #ca8a04;"
        dark_badge_color = "#facc15"
    else:
        badge_style = "background: rgba(220, 38, 38, 0.2); color: #dc2626; border: 1px solid #dc2626;"
        dark_badge_color = "#f87171"

    html = f"""
    <div class="result-card">
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="color: var(--body-text-color-subdued); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; 
                        font-weight: 700; margin-bottom: 15px;">Analysis Results</div>
            
            <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
                <div style="background: rgba(59, 130, 246, 0.1); padding: 10px 20px; border-radius: 15px; border: 2px solid #3b82f6;">
                    <div style="font-size: 0.75rem; color: #3b82f6; font-weight: 600; margin-bottom: 3px;">🔍 YOLO Detection</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #3b82f6;">{yolo_status}</div>
                    {f'<div style="font-size: 0.8rem; color: #3b82f6; margin-top: 3px;">{num_defects} region(s)</div>' if num_defects > 0 else ''}
                </div>
                <div style="background: rgba(168, 85, 247, 0.1); padding: 10px 20px; border-radius: 15px; border: 2px solid #a855f7;">
                    <div style="font-size: 0.75rem; color: #a855f7; font-weight: 600; margin-bottom: 3px;">🤖 SVM Classification</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #a855f7;">Classified</div>
                </div>
            </div>
            
            <div style="font-size: 3rem; font-weight: 800; margin: 15px 0; color: var(--body-text-color); letter-spacing: -1px;">
                {defect_type}
            </div>
            
            <style>
                .dark .dynamic-badge {{ color: {dark_badge_color} !important; border-color: {dark_badge_color} !important; }}
            </style>
            <div class="confidence-badge dynamic-badge" style="{badge_style}">
                Confidence: {confidence}
            </div>
        </div>
        
        <div style="border-top: 2px solid var(--border-color-primary); padding-top: 20px;">
            <h4 style="margin-bottom: 20px; color: var(--body-text-color); font-weight: 700; font-size: 1.1rem;">📈 Top 3 Details:</h4>
    """
    
    for i, pred in enumerate(top3):
        width = float(pred['Xác suất'].strip('%'))
        bar_color = "#667eea" if i == 0 else "#764ba2" if i == 1 else "#94a3b8"
        
        html += f"""
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
    
    html += "</div></div>"
    return html
