import gradio as gr
import os, traceback

from lib.PoseViewModel import ProcessingViewModel 
from lib.Pose3dDictEstim import Pose3dDictEstim
from lib.PoseTimeseries import PoseTimeseries
from lib.Vector3dTimeseries import Vector3dTimeseries
from lib.Vector2dTimeseries import Vector2dTimeseries
from lib.BagData import BagData
from views.utils import success_text, error_text

from config import DATA_BASE_DIR


class ProcessView:
    @staticmethod
    def get_file_selector_choices():
        return BagData.list_parsed(DATA_BASE_DIR)
    
    def __init__(self, app):
        self.complete_signal = gr.State(False)
        self.file_selector = None
        
        vm = ProcessingViewModel(app)
        gr.Markdown("## 2. Create set of pose pivot timeseries")
        gr.Markdown("This step iterates over the parsed frame data. It runs a pose detection model on each frame, and saves the pose pivots with apropriate depth value as a 3d coordiante. These 3d coordinates of each body part (pivot) are saved as a timeseries.")
        gr.Markdown("It takes about **40-70min** for **each minute of footage**. Turn caching on to use saved progress.")
        
        with gr.Row():
            file_selector = gr.Dropdown(
                ProcessView.get_file_selector_choices(),
                label="Select parsed .bag data"
            )
            
            with gr.Column():
                with gr.Row():
                    use_cache = gr.Checkbox(True, label="Use cached progress")
                    gr.Markdown("Progress is saved to the disk. Disable to ignore progress and re-run from scratch.")
                process_btn = gr.Button("Detect pose")
            
        output = gr.Markdown("", height=50)
            
        @process_btn.click(inputs=[file_selector, use_cache], outputs=[output, self.complete_signal])
        def _on_process_click(file_name, use_cache, progress=gr.Progress(track_tqdm=True)):
            try:
                if file_name:
                    data_path = os.path.join(DATA_BASE_DIR, file_name)
                    pose3d_array, error = Pose3dDictEstim.from_unpacked_bag(data_path, use_cache)

                    if error:
                        return error_text(error), False

                    pose_timeseries = PoseTimeseries.get(pose3d_array)
                    PoseTimeseries.save(data_path, pose_timeseries, suffix="_estim")

                    vector3d_timeseries = Vector3dTimeseries.from_pose_timeseries(pose_timeseries)
                    Vector3dTimeseries.save(data_path, vector3d_timeseries, suffix="_estim")

                    vector2d_timeseries = Vector2dTimeseries.from_pose_timeseries(pose_timeseries)
                    Vector2dTimeseries.save(data_path, vector2d_timeseries, suffix="_estim")
                    
                    return success_text(f"Processed successfully."), True
                return "No data selected", False
            except Exception as e:
                traceback.print_exc()
                return error_text(str(e))
    
        self.file_selector = file_selector
        self.output = output