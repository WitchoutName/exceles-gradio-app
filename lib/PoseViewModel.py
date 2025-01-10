import numpy as np
import gradio as gr
from lib.BagData import BagData
from lib.Filter import Filter
from lib.PoseTimeseries import PoseTimeseries


class ProcessingViewModel:
    def __init__(self, app):
        self.app = app


class VizuViewModel:
    def __init__(self, app):
        self.app = app
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