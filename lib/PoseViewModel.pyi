import numpy as np
import gradio as gr
import os
import json
from enum import Enum
from lib.BagData import BagData
from lib.Filter import Filter
from lib.PoseTimeseries import PoseTimeseries
from lib.Vector2dTimeseries import Vector2dTimeseries
from lib.Vector3dTimeseries import Vector3dTimeseries


class ProcessingViewModel:
    def __init__(self, app):
        self.app = app


class PivotAxisName(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    Vector2d = "XY 2D Vector magnitude"
    Vector3d = "XYZ 3D Vector magnitude"

from gradio.events import Dependency

class SessionTag(gr.Label):
    pass
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer


class Session:
    def __init__(self, app, session_tag):
        self.app = app
        self.processing = ProcessingViewModel(app)
        self.vizu = VizuViewModel(app)
    
    # @staticmethod
    # def inject_tag():
    #     tag = gr.Label(visible=False)
    #     @tag.on



class VizuViewElements:
    def __init__(self):
        self.filter_name = None
        self.pivot_selector = []
        self.min_max = []
        self.use_filter = []
        

class VizuViewModel:
    def __init__(self, app):
        self.app = app
        self.elements = VizuViewElements()
        self.bag_data = gr.State(None)
        self.filter = gr.State(None)
        self.filter_params = gr.State([])
        self.pivot = [gr.State([[], [], []]), gr.State([[], [], []])]
        self.x_axis = [gr.State([]), gr.State([])]
        self.y_axis = [gr.State([]), gr.State([])]
        self.z_axis = [gr.State([]), gr.State([])]

    def set_bag_data(self, data_path):
        return BagData(data_path)

    def set_pivot(self, bag_data, part_name):
        pose_dict_key, index = PoseTimeseries.get_body_part_object_path(part_name)
        pivot = bag_data.pose_timeseries[pose_dict_key][index]
        return pivot
    
    def on_is_filtered_change_factory(axis):
        def _on_is_filtered_change(is_filtered, pivot, filter_fn=lambda x: x, filter_params=[], start_frame=0, end_frame=100):
            pivot_interval = pivot[axis][start_frame:end_frame]
            if not filter_fn: return pivot_interval
            fn = lambda x: filter_fn(x, *filter_params)
            return PoseTimeseries.filter_axis(pivot_interval, fn) if is_filtered else pivot_interval
        return _on_is_filtered_change
    
    def get_data_dict(self, pivot_axis_name: PivotAxisName):
        def axis_name_to_timeseries(i):
            return {
                PivotAxisName.X: self.x_axis[i].value,
                PivotAxisName.Y: self.y_axis[i].value,
                PivotAxisName.Z: self.z_axis[i].value,
                PivotAxisName.Vector2d: Vector2dTimeseries.pose_to_vector_length(np.array([[self.x_axis[i].value, self.y_axis[i].value, np.zeros_like(self.x_axis[i].value)]]))[0],
                PivotAxisName.Vector3d: Vector3dTimeseries.pose_to_vector_length(np.array([[self.x_axis[i].value, self.y_axis[i].value, self.z_axis[i].value]]))[0]
            }[pivot_axis_name]

        def apply_fft(data):
            fourier = np.fft.fft(data)
            half_n = data.shape[0] // 2
            return np.abs(fourier[:half_n])
        
        def export_pivot_data(i):
             data = axis_name_to_timeseries(i)
             return {
                "filtered_axis": [f.value for f in self.elements.use_filter[0]],
                "start_frame": self.elements.min_max[0][0].value,
                "end_frame": self.elements.min_max[0][1].value,
                "pivot": self.elements.pivot_selector[0].value,
                "timeseries": data,
                "fft_magnitude": apply_fft(data) if data else None
            }
        
        data = {
            "bag_file": self.bag_data.value.bag_name if self.bag_data.value else None,
            "filter": {
                "name": self.elements.filter_name.value,
                "params": self.filter_params.value
            },
            "pivot_axis_name": pivot_axis_name.value,
            "pivot_1": export_pivot_data(0),
            "pivot_2": export_pivot_data(1)
        }
        
        print(data)
        return data
        
        # Example of the data
        data = {
            "bag_file": "example.bag",                # Name of the bag file
            "filter": {                               # Filter used
                "name": "Gaussian",                   # Name of the filter
                "params": [2]                         # Parameters of the filter
            },
            "pivot_axis_name": "X",                   # Axis of the pivot (X, Y, Z, XY 2D Vector magnitude, XYZ 3D Vector magnitude)
            "pivot_1": {
                "filtered_axis": [True, False, True], # True if axis was filtered, False otherwise
                "start_frame": 0,                     # Start frame of the data
                "end_frame": 100,                     # End frame of the data
                "pivot": "Neck",                      # Name of the pivot (body part)
                "timeseries": [0.1, 0.2, 0.3, 0.4],   # Timeseries data
                "fft_magnitude": [0.01, 0.02]         # FFT of the timeseries     
            },
            "pivot_2": {
                "filtered_axis": [True, False, True], # True if axis was filtered, False otherwise
                "start_frame": 0,                     # Start frame of the data
                "end_frame": 100,                     # End frame of the data
                "pivot": "Neck",                      # Name of the pivot (body part)
                "timeseries": [0.1, 0.2, 0.3, 0.4],   # Timeseries data
                "fft_magnitude": [0.01, 0.02]         # FFT of the timeseries     
            },
        }
        
    def export_json(self, pivot_axis_name: PivotAxisName):
        data = self.get_data_dict(pivot_axis_name)
        file_name = f"{self.bag_data.value.bag_name if self.bag_data.value else ''}_{pivot_axis_name.value}.json"
        file_path = os.path.join("/tmp", file_name)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return file_path

    
    def export_data(self, pivot_axis_name: PivotAxisName, format="json"):
        if format == "json":
            return self.export_json(pivot_axis_name)
        