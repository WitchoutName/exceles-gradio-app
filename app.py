import gradio as gr
import os, sys

from config import DATA_BASE_DIR

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'comfyui_controlnet_aux', 'src'))

from views.ParseView import ParseView
from views.ProcessView import ProcessView
from views.VisualizeView import VisualizeView

with gr.Blocks() as app:
    with gr.Tab("Parse"):
        parse_view = ParseView(app)
        parse_output = parse_view.output
        parse_complete_signal = parse_view.complete_signal
    with gr.Tab("Detect pose"):
        process_view = ProcessView(app)
        process_file_sepector = process_view.file_selector
        process_output = process_view.output
        process_complete_signal = process_view.complete_signal
    with gr.Tab("Visualize"):
        visu_view = VisualizeView(app)
        visu_file_sepector = visu_view.file_selector
        
    @parse_complete_signal.change(inputs=[parse_complete_signal], outputs=[process_file_sepector, parse_complete_signal])
    def parse_output_change(complete_signal):
        print("parse_output_change", complete_signal)
        if complete_signal:
            return gr.update(choices=ProcessView.get_file_selector_choices()), False
        return gr.update(), gr.update()
    
    @process_complete_signal.change(inputs=[process_complete_signal], outputs=[visu_file_sepector, process_complete_signal])
    def process_output_change(complete_signal):
        if complete_signal:
            return gr.update(choices=VisualizeView.get_file_selector_choices()), False
        return gr.update(), gr.update()
            
app.launch(share=True, allowed_paths=[DATA_BASE_DIR])