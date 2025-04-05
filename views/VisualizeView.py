import gradio as gr
import os, traceback
import numpy as np
from enum import Enum

from lib.BagData import BagData
from lib.PoseTimeseries import PoseTimeseries
from lib.Vector2dTimeseries import Vector2dTimeseries
from lib.Vector3dTimeseries import Vector3dTimeseries
from lib.PoseViewModel import VizuViewModel, PivotAxisName
from lib.Filter import Filter
from views.utils import format_video_info

from config import DATA_BASE_DIR

import gradio.events as Events


class VisualizeView:
    @staticmethod
    def get_file_selector_choices():
        return ["Select"] + BagData.list_processed(DATA_BASE_DIR)
    

    def __init__(self, app: gr.Blocks):
        self.file_selector = None
        
        vm = VizuViewModel(app)
        gr.Markdown("### Select .bag file")
        with gr.Row():
            with gr.Column():
                file_selector = gr.Dropdown(
                    VisualizeView.get_file_selector_choices(),
                    label="Select .bag file"
                )
                
                @file_selector.change(inputs=[file_selector, vm.filter], outputs=[vm.bag_data, vm.filter])
                def _on_file_change(file_name, filter_fn):
                    print(file_name)
                    if filter_fn == None:
                        filter_fn = lambda x: x
                        
                    if file_name == "Select":
                        return [None, filter_fn]
                    data_path = os.path.join(DATA_BASE_DIR, file_name+".bag_data")
                    print(data_path)
                    return [vm.set_bag_data(data_path), filter_fn]  
                

            with gr.Column():
                # color video of the .bag
                video = gr.Video(render=True)
                
                def _on_bag_data_change__video(bag_data, progress=gr.Progress(track_tqdm=True)):
                    return bag_data.get_video()            
                vm.bag_data.change(_on_bag_data_change__video, [vm.bag_data], [video])

                video_info = gr.Markdown(format_video_info())
                
                @vm.bag_data.change(inputs=[vm.bag_data], outputs=[video_info])
                def _on_bag_data_change__video_info(bag_data):
                    print("bag_data changed")
                    return format_video_info(bag_data.get_frame_count(), bag_data.duration)



        gr.Markdown("### Define a smoothing filter")
        with gr.Row():
            class FilterType(Enum):
                NONE = "None"
                MOVING_AVERAGE = "Moving Average"
                GAUSSIAN = "Gaussian"
                SAVITZKY_GOLAY = "Savitzky-Golay"

            filter_type = gr.Radio([x.value for x in FilterType], label="Smoothing Filter", value=FilterType.NONE.value)
            

            window_input = gr.Number(label="Window Size (%) for Moving Average", value=5, visible=False)
            sigma_input = gr.Number(label="Sigma for Gaussian Filter", value=2, visible=False)
            window_input_sva = gr.Number(label="Window Length for Savitzky-Golay Filter (Odd Number)", value=11, visible=False)
            polyorder_input = gr.Number(label="Polynomial Order for Savitzky-Golay Filter", value=2, visible=False)

            def _on_filter_type_change_factory(target_filter):
                def _on_filter_type_change(filter_type):
                    if filter_type == target_filter.value:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                return _on_filter_type_change

            filter_type.change(_on_filter_type_change_factory(FilterType.MOVING_AVERAGE), inputs=[filter_type], outputs=[window_input])
            filter_type.change(_on_filter_type_change_factory(FilterType.GAUSSIAN), inputs=[filter_type], outputs=[sigma_input])
            filter_type.change(_on_filter_type_change_factory(FilterType.SAVITZKY_GOLAY), inputs=[filter_type], outputs=[window_input_sva])
            filter_type.change(_on_filter_type_change_factory(FilterType.SAVITZKY_GOLAY), inputs=[filter_type], outputs=[polyorder_input])


            # @filter_type.change(inputs=[filter_type, window_input, sigma_input, window_input_sva, polyorder_input], outputs=[vm.filter, vm.filter_params])
            def _on_filter_type_change(filter_type, window_input, sigma_input, window_input_sva, polyorder_input):
                if filter_type == FilterType.MOVING_AVERAGE.value:
                    return [Filter.moving_average, [window_input / 100]]
                elif filter_type == FilterType.GAUSSIAN.value:
                    return [Filter.gaussian, [sigma_input]]
                elif filter_type == FilterType.SAVITZKY_GOLAY.value:
                    return [Filter.savgol, [window_input_sva, polyorder_input]]
                return [lambda x: x, []]

            gr.on(
                triggers=[filter_type.change, window_input.change, sigma_input.change, window_input_sva.change, polyorder_input.change],
                fn=_on_filter_type_change,
                inputs=[filter_type, window_input, sigma_input, window_input_sva, polyorder_input], 
                outputs=[vm.filter, vm.filter_params]
            )
            

        with gr.Row():
            def part_config(pi):
                with gr.Column():
                    x_axis, y_axis, z_axis = vm.x_axis[pi], vm.y_axis[pi], vm.z_axis[pi]
                    gr.Markdown(f"## Body/Hand Part {pi+1}")
                    part_selector = gr.Dropdown(
                        ["Select"] + PoseTimeseries.all_parts_list,
                        value="Select",
                        label="Select Body/Hand Part"
                    )
                    
                    def _on_part_change(bag_data, part_name):
                        print(part_name)
                        if part_name == "Select":
                            return [np.array([[], [], []]), *[np.array([])]*3]
                        pivot = vm.set_pivot(bag_data, part_name)
                        return [pivot, pivot[0], pivot[1], pivot[2]]
                    
                    gr.on(
                        triggers=[part_selector.change, vm.bag_data.change], 
                        fn=_on_part_change,
                        inputs=[vm.bag_data, part_selector],
                        outputs=[vm.pivot[pi], x_axis, y_axis, z_axis]
                    )
                    
                    
                    gr.Markdown("### Limit the interval of frames to process")
                    with gr.Row():
                        with gr.Column(min_width=540):
                            start_frame = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Start Frame")
                            end_frame = gr.Slider(minimum=0, maximum=100, step=1, value=100, label="End Frame")
                            
                            bag_data = vm.bag_data
                            @bag_data.change(inputs=[bag_data], outputs=[start_frame, end_frame])
                            def _on_frame_interval_change(bag_data):
                                if bag_data:
                                    length = bag_data.get_frame_count()
                                    return [gr.update(maximum=length, value=0), gr.update(maximum=length, value=length)]
                                return [gr.update(), gr.update()]
                            
                        full_interval = gr.Button("Full Interval")
                        @full_interval.click(inputs=[bag_data], outputs=[start_frame, end_frame])
                        def _on_full_interval_click(bag_data):
                            if bag_data:
                                length = bag_data.get_frame_count()
                                return [0, length]
                            return [0, 100]
                    
        
                    gr.Markdown("#### X Axis")
                    with gr.Row(height=55):
                        x_is_filtered = gr.Checkbox(label="Use Filter (X)")
                        if pi == 1:
                            x_data_file = gr.File(label="Download X Axis", file_types=["json"])
                            def _on_x_axis_change():
                                return vm.export_data(PivotAxisName.X)
                            gr.on(triggers=[e.change for e in vm.x_axis], fn=_on_x_axis_change, outputs=[x_data_file])


                    x_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    x_fft_plot = gr.Image()
                    def _on_x_axis_change(x_axis):
                        return [
                            PoseTimeseries.plot_pivot_axes(x_axis, "X"),
                            PoseTimeseries.plot_pivot_axes_fft(x_axis, "X") if len(x_axis) > 0 else None
                        ]
                    x_axis.change(_on_x_axis_change, inputs=[x_axis], outputs=[x_plot, x_fft_plot])
                                        

                    gr.Markdown("#### Y Axis")
                    with gr.Row(height=55):
                        y_is_filtered = gr.Checkbox(label="Use Filter (Y)")
                        if pi == 1:
                            y_data_file = gr.File(label="Download Y Axis", file_types=["json"])
                            def _on_y_axis_change():
                                return vm.export_data(PivotAxisName.Y)
                            gr.on(triggers=[e.change for e in vm.y_axis], fn=_on_y_axis_change, outputs=[y_data_file])


                    y_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    y_fft_plot = gr.Image()
                    def _on_y_axis_change(y_axis):
                        return [
                            PoseTimeseries.plot_pivot_axes(y_axis, "Y"),
                            PoseTimeseries.plot_pivot_axes_fft(y_axis, "Y") if len(y_axis) > 0 else None
                        ]
                    y_axis.change(_on_y_axis_change, inputs=[y_axis], outputs=[y_plot, y_fft_plot])

                    gr.Markdown("#### Z Axis")
                    with gr.Row(height=55):
                        z_is_filtered = gr.Checkbox(label="Use Filter (Z)")
                        if pi == 1:
                            z_data_file = gr.File(label="Download Z Axis", file_types=["json"])
                            def _on_z_axis_change():
                                return vm.export_data(PivotAxisName.Z)
                            gr.on(triggers=[e.change for e in vm.z_axis], fn=_on_z_axis_change, outputs=[z_data_file])


                    z_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    z_fft_plot = gr.Image()
                    def _on_z_axis_change(z_axis):
                        return [
                            PoseTimeseries.plot_pivot_axes(z_axis, "Z"),
                            PoseTimeseries.plot_pivot_axes_fft(z_axis, "Z") if len(z_axis) > 0 else None
                        ]
                    z_axis.change(_on_z_axis_change, inputs=[z_axis], outputs=[z_plot, z_fft_plot])                        

                    def _on_is_filtered_change_factory(axis):
                        def _on_is_filtered_change(is_filtered, pivot, filter_fn=lambda x: x, filter_params=[], start_frame=0, end_frame=100):
                            pivot_interval = pivot[axis][start_frame:end_frame]
                            if not filter_fn: return pivot_interval
                            fn = lambda x: np.array(filter_fn(x, *filter_params))
                            return PoseTimeseries.filter_axis(pivot_interval, fn) if is_filtered else pivot_interval
                        return _on_is_filtered_change
                    
                    gr.on(
                        triggers=[x_is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change, start_frame.change, end_frame.change], 
                        fn=_on_is_filtered_change_factory(0),
                        inputs=[x_is_filtered, vm.pivot[pi], vm.filter, vm.filter_params, start_frame, end_frame],
                        outputs=[x_axis]
                    )
                    
                    gr.on(
                        triggers=[y_is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change, start_frame.change, end_frame.change], 
                        fn=_on_is_filtered_change_factory(1),
                        inputs=[y_is_filtered, vm.pivot[pi], vm.filter, vm.filter_params, start_frame, end_frame],
                        outputs=[y_axis]
                    )
                    
                    gr.on(
                        triggers=[z_is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change, start_frame.change, end_frame.change], 
                        fn=_on_is_filtered_change_factory(2),
                        inputs=[z_is_filtered, vm.pivot[pi], vm.filter, vm.filter_params, start_frame, end_frame],
                        outputs=[z_axis]
                    )
                        
                    
                    gr.Markdown("## XY 2D Vector magnitude")
                    with gr.Row(height=55):
                        if pi == 1:
                            v2_data_file = gr.File(label="Download XY Vector magnitude", file_types=["json"])
                            def _on_v2_change(x_axis, y_axis):
                                return vm.export_data(PivotAxisName.Vector2d)
                            gr.on(
                                triggers=[x_axis.change, y_axis.change], 
                                fn=_on_v2_change,
                                inputs=[x_axis, y_axis], 
                                outputs=[v2_data_file]
                            )

                    v2_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    v2_fft_plot = gr.Image()
                    
                    
                    gr.Markdown("## XYZ 3D Vector magnitude")
                    with gr.Row(height=55):
                        if pi == 1:
                            v3_data_file = gr.File(label="Download XYZ Vector magnitude", file_types=["json"])
                            def _on_v3_change(x_axis, y_axis, z_axis):
                                return vm.export_data(PivotAxisName.Vector3d)
                            gr.on(
                                triggers=[x_axis.change, y_axis.change], 
                                fn=_on_v3_change,
                                inputs=[x_axis, y_axis, z_axis], 
                                outputs=[v3_data_file]
                            )
                        
                    v3_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    v3_fft_plot = gr.Image()
                    
                    def update_v2_plot(x, y):
                        if not (x.size == y.size):
                            return [None, None]
                        pivot = np.array([x, y, np.zeros_like(x)])
                        mock_pose_timeseries = np.array([pivot])
                        vector2d = Vector2dTimeseries.pose_to_vector_length(mock_pose_timeseries)[0]
                        return [
                            Vector2dTimeseries.plot_pivot(vector2d),
                            Vector2dTimeseries.plot_pivot_fft(vector2d, "vector XY magnitude") if len(vector2d) > 0 else None
                        ]
                    gr.on(
                        triggers=[*[e.change for e in vm.x_axis], *[e.change for e in vm.y_axis]], 
                        fn=update_v2_plot,
                        inputs=[x_axis, y_axis], 
                        outputs=[v2_plot, v2_fft_plot]
                    )
                    
                    def update_v3_plot(x_axis, y_axis, z_axis):
                        if not (x_axis.size == y_axis.size == z_axis.size):
                            return [None, None]
                        pivot = np.array([x_axis, y_axis, z_axis])
                        mock_pose_timeseries = np.array([pivot])
                        vector3d = Vector3dTimeseries.pose_to_vector_length(mock_pose_timeseries)[0]
                        return [
                            Vector3dTimeseries.plot_pivot(vector3d, rank="3D"),
                            Vector3dTimeseries.plot_pivot_fft(vector3d, "vector XYZ magnitude") if len(vector3d) > 0 else None
                        ]
                    gr.on(
                        triggers=[*[e.change for e in vm.x_axis], *[e.change for e in vm.y_axis], *[e.change for e in vm.z_axis]], 
                        fn=update_v3_plot,
                        inputs=[x_axis, y_axis, z_axis], 
                        outputs=[v3_plot, v3_fft_plot]
                    )
                
                vm.comp_state.filter_name = filter_type
                vm.comp_state.pivot_selector.append(part_selector)
                vm.comp_state.min_max.append([start_frame, end_frame])
                vm.comp_state.use_filter.append([x_is_filtered, y_is_filtered, z_is_filtered])
        
            for part_index in range(2):
                part_config(part_index)
                
                
        self.file_selector = file_selector
        vm.session.inject_tag()
        vm.session.update()
        vm.comp_state.transform_components()