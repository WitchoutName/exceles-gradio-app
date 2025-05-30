import numpy as np
import gradio as gr
from collections import deque
from gradio.components.base import Component
import os
import json
from enum import Enum
from lib.BagData import BagData
from lib.Filter import Filter
from lib.PoseTimeseries import PoseTimeseries
from lib.Vector2dTimeseries import Vector2dTimeseries
from lib.Vector3dTimeseries import Vector3dTimeseries
from lib.Session import Session


class ProcessingViewModel:
    def __init__(self, app):
        self.app = app


class PivotAxisName(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"
    Vector2d = "XY 2D Vector magnitude"
    Vector3d = "XYZ 3D Vector magnitude"

    
    
class ComponentStateProxy:
    @staticmethod
    def initialize_state(component: Component):
        """
        Initialize the state for a given component.
        
        Args:
            component (Component): The component to initialize the state for.
        
        Returns:
            gr.State: The initialized state.
        """
        state = gr.State(None)
        # @component.change(inputs=[component], outputs=[state])
        def _on_component_change(value):
            return value

        gr.on(
            triggers=[component.change], 
            fn=_on_component_change, 
            inputs=[component], outputs=[state]
        )
    
        return state
    
    
        
    def transform_components(self):
        """
        Transform the components in the current instance by hooking up their states.
        """
        queue = deque()
        
        for key, value in self.__dict__.items():
            if isinstance(value, Component):
                setattr(self, key, ComponentStateProxy.initialize_state(value))
            else:
                queue.append(value)
        
        while queue:
            obj = queue.popleft()
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, Component):
                        obj[key] = ComponentStateProxy.initialize_state(value)
                    else:
                        queue.append(value)
            elif isinstance(obj, (list, tuple, set)):
                for i, value in enumerate(obj):
                    if isinstance(value, Component):
                        obj[i] = ComponentStateProxy.initialize_state(value)
                    else:
                        queue.append(value)
            else:
                raise ValueError(f"Unsupported type: {type(obj)}")
                

class VizuViewComponentState(ComponentStateProxy):
    def __init__(self):
        self.filter_name = None
        self.pivot_selector = []
        self.min_max = []
        self.use_filter = []
        

class VizuViewModel:
    def __init__(self, app):
        self.app = app
        self.comp_state = VizuViewComponentState()
        self.session = Session(app)
        self.bag_data = gr.State(None)
        self.filter = gr.State(None)
        self.filter_params = gr.State([])
        self.pivot = [gr.State(np.array([[], [], []])), gr.State(np.array([[], [], []]))]
        self.x_axis = [gr.State(np.array([])), gr.State(np.array([]))]
        self.y_axis = [gr.State(np.array([])), gr.State(np.array([]))]
        self.z_axis = [gr.State(np.array([])), gr.State(np.array([]))]

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
        get_value = self.session.get_value
        def axis_name_to_timeseries(i):
            x_axis, y_axis, z_axis =  [np.array(get_value(x)) for x in (self.x_axis[i], self.y_axis[i], self.z_axis[i])]
            if x_axis.size == y_axis.size == z_axis.size:
                vector2d = Vector2dTimeseries.pose_to_vector_length(np.array([[x_axis, y_axis, np.zeros_like(x_axis)]]))[0]
                vector3d = Vector3dTimeseries.pose_to_vector_length(np.array([[x_axis, y_axis, z_axis]]))[0]
            else:
                vector2d = vector3d = np.array([])
                
            return {
                PivotAxisName.X: x_axis,
                PivotAxisName.Y: y_axis,
                PivotAxisName.Z: z_axis,
                PivotAxisName.Vector2d: vector2d,
                PivotAxisName.Vector3d: vector3d
            }[pivot_axis_name]
            
        def apply_fft(data: np.ndarray) -> np.ndarray:
            fourier = np.fft.fft(data)
            half_n = data.shape[0] // 2
            return np.abs(fourier[:half_n])
        
        def export_pivot_data(i):
             data: np.ndarray = np.array(axis_name_to_timeseries(i))
             return {
                "filtered_axis": [get_value(axis) or False for axis in self.comp_state.use_filter[0]],
                "start_frame": get_value(self.comp_state.min_max[0][0]),
                "end_frame": get_value(self.comp_state.min_max[0][1]),
                "pivot": get_value(self.comp_state.pivot_selector[0]),
                "timeseries": data.tolist(),
                "fft_magnitude": apply_fft(data).tolist() if data.size > 0 else []
            }
        
        bag_data = get_value(self.bag_data)
        data = {
            "bag_file": bag_data.bag_name if bag_data else None,
            "filter": {
                "name": get_value(self.comp_state.filter_name),
                "params": get_value(self.filter_params)
            },
            "pivot_axis_name": pivot_axis_name.value,
            "pivot_1": export_pivot_data(0),
            "pivot_2": export_pivot_data(1)
        }
        
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
        file_name = f"{data.get('bag_file')}_{pivot_axis_name.value}.json"
        file_path = os.path.join("/tmp", file_name)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return file_path

    
    def export_data(self, pivot_axis_name: PivotAxisName, format="json"):
        if format == "json":
            return self.export_json(pivot_axis_name)
        