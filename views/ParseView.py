import gradio as gr
import os, traceback

from lib.BagFile import BagFile
from lib.BagData import BagData
from views.utils import success_text, error_text, format_video_info

from config import BAG_UPLOAD_DIR, DATA_BASE_DIR


class ParseView:
    def __init__(self, app):
        self.complete_signal = gr.State(False)
        
        gr.Markdown("## 1. Parse raw file on system")
        gr.Markdown("This step will parse the selected .bag file on the server and save the color map, depth map and confidence map to the file system.")
        gr.Markdown("It takes about **5min** for **each minute of footage**.")
        
        with gr.Row():
            with gr.Column():
                file_selector = gr.Dropdown(
                    BagFile.list_raw_files(BAG_UPLOAD_DIR),
                    label="Select a .bag file to parse"
                )
                gr.Markdown("\n\n\n\n\n"), gr.Markdown(), gr.Markdown(), gr.Markdown(), gr.Markdown(), gr.Markdown()
                with gr.Row():
                    input_feedback = gr.Markdown(f"`Selected: {file_selector.value}`")
                    parse_file_btn = gr.Button("Unpack selected .bag file")
                    

            file_upload = gr.File(label="Upload a new .bag file")
            
            
            @file_upload.change(inputs=[file_upload, file_selector], outputs=[input_feedback, file_selector, file_upload])
            def _on_file_upload(temp_path, file_selector):
                if temp_path:
                    file_name, message = BagFile.upload_file(temp_path, BAG_UPLOAD_DIR)
                    if file_name:
                        return [f"`{message}`", gr.update(choices=BagFile.list_raw_files(BAG_UPLOAD_DIR), value=file_name), None]
                    else:
                        return [error_text(message), file_selector, None]
                return [gr.update(), gr.update(), None]
            
        @file_selector.change(inputs=[file_selector], outputs=[input_feedback])
        def _on_file_change(file_name):
            return f"`Selected: {file_name}`"

        stats_info = gr.Markdown("", height=120)
        output = gr.Markdown(".", height=50)
        
        
        @parse_file_btn.click(inputs=[file_selector], outputs=[stats_info])
        def _on_parse_file_click(file_name, progress=gr.Progress(track_tqdm=True)):
            if file_name:
                bag_file = os.path.join(BAG_UPLOAD_DIR, file_name)
                frame_fount, duration, fps = BagFile.get_stats(bag_file)
                return format_video_info(frame_fount, duration)
            return ""
        
        @parse_file_btn.click(inputs=[file_selector], outputs=[output, self.complete_signal])
        def _on_parse_file_click(file_name, progress=gr.Progress(track_tqdm=True)):
            try:
                if file_name:
                    bag_file = os.path.join(BAG_UPLOAD_DIR, file_name)
                    output_dir = os.path.join(DATA_BASE_DIR, file_name+"_data")
                    BagFile.parse(bag_file, output_dir, True)
                    return success_text(f"File \"{file_name}\" parsed successfully. Avaliable as \"{file_name+'_data'}\" in next steps."), True
                return "No file selected", False
            except Exception as e:
                return error_text(str(e))
            
        self.output = output
            
        