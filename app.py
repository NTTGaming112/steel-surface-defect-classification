"""
Main application file - Steel Defect Detection AI
"""
import sys
import gradio as gr

# Import utils first to make SiftBowExtractor available for pickle
import utils  # Required for unpickling SIFT extractor

from src.core.models import model_manager
from src.core.prediction import predict_defect
from src.ui.ui_components import (
    CUSTOM_CSS,
    get_header_html,
    get_info_html,
    get_waiting_html,
    get_cleared_html,
    get_stats_html,
    get_footer_html,
    generate_result_html
)

# Load all models
model_manager.load_models()

def predict_wrapper(image):
    """Wrapper function for prediction with HTML rendering"""
    json_res, proc_img, text_res = predict_defect(image)
    
    if json_res is None or isinstance(json_res, str):
        error_html = f"<div style='color: var(--error-text-color); padding: 20px; background: var(--background-fill-secondary); border-radius: 8px; border: 1px solid var(--error-border-color);'>{text_res}</div>"
        return error_html, None, None
    
    result_html = generate_result_html(json_res)
    return result_html, proc_img, json_res

def clear_interface():
    """Clear all inputs and outputs"""
    return None, get_cleared_html(), None, None

# Build Gradio interface
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.slate,
    font=("Poppins", "sans-serif")
)

with gr.Blocks(title="Steel Defect AI", theme=theme, css=CUSTOM_CSS) as demo:
    
    # Header
    with gr.Row(elem_classes="header-container"):
        with gr.Column():
            gr.HTML(get_header_html())

    # Main content
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=5):
            gr.HTML('<div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem;">📤 Upload Image</div>')
            
            img_input = gr.Image(
                type="pil",
                label="",
                height=420,
                sources=["upload", "clipboard"],
                elem_id="img_input"
            )
            
            gr.HTML(get_info_html())
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg")
                predict_btn = gr.Button("🚀 Analyze", variant="primary", size="lg", elem_id="predict-btn")

        # Right column - Output
        with gr.Column(scale=6):
            gr.HTML('<div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem;">📊 Analysis Results</div>')
            
            with gr.Tabs():
                with gr.TabItem("🎯 Dashboard"):
                    result_html = gr.HTML(value=get_waiting_html())
                
                with gr.TabItem("🔍 YOLO Detection"):
                    processed_output = gr.Image(label="", interactive=False, height=400)
                
                with gr.TabItem("📋 Raw Data"):
                    output_json = gr.JSON(label="")

    # Statistics cards
    gr.HTML(get_stats_html())
    
    # Footer
    gr.HTML(get_footer_html())

    # Event handlers
    predict_btn.click(
        fn=predict_wrapper,
        inputs=img_input,
        outputs=[result_html, processed_output, output_json]
    )
    
    clear_btn.click(
        fn=clear_interface,
        inputs=None,
        outputs=[img_input, result_html, processed_output, output_json]
    )

if __name__ == "__main__":
    print("🚀 Starting Steel Defect Detection App...")
    demo.launch(share=False)
