import gradio as gr
import os, traceback
import numpy as np
from enum import Enum

from lib.BagData import BagData
from lib.PoseTimeseries import PoseTimeseries
from lib.Vector2dTimeseries import Vector2dTimeseries
from lib.Vector3dTimeseries import Vector3dTimeseries
from lib.PoseViewModel import VizuViewModel 
from lib.Filter import Filter
from views.utils import format_video_info

from config import DATA_BASE_DIR

class VisualizeView:
    @staticmethod
    def get_file_selector_choices():
        return ["Select"] + BagData.list_processed(DATA_BASE_DIR)

    def __init__(self, app):
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
                    return format_video_info(bag_data.frame_count, bag_data.duration)



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
                    gr.Markdown(f"## Body/Hand Part {pi+1}")
                    part_selector = gr.Dropdown(
                        ["Select"] + PoseTimeseries.all_parts_list,
                        value="Select",
                        label="Select Body/Hand Part"
                    )
                    
                    @part_selector.change(inputs=[vm.bag_data, part_selector], outputs=[vm.pivot[pi], vm.x_axis[pi], vm.y_axis[pi], vm.z_axis[pi]])
                    def _on_part_change(bag_data, part_name):
                        print(part_name)
                        if part_name == "Select":
                            return [np.array([[], [], []]), *[np.array()]*3]
                        pivot = vm.set_pivot(bag_data, part_name)
                        return [pivot, pivot[0], pivot[1], pivot[2]]
                    
                    def _on_bag_data_change_part_selector(bag_data, progress=gr.Progress(track_tqdm=True)):
                        if bag_data:
                            bag_data.process_frames()
                        return gr.update(visible=bag_data)
                    # vm.bag_data.change(_on_bag_data_change_part_selector, vm.bag_data, part_selector)
                    
                    
                    # for [vm_axis, axis_name] in [[vm.x_axis[pi], "X"], [vm.y_axis[pi], "Y"], [vm.z_axis[pi], "Z"]]:
                    #     gr.Markdown(f"### {axis_name} Axis")
                    #     is_filtered = gr.Checkbox(label=f"Use Filter ({axis_name})")
                    #     plot = gr.Image()
                    #     gr.Markdown(f"###### FFT")
                    #     fft_plot = gr.Image()
                    #     def _on_axis_change(axis):
                    #         print(f"Axis Name: {axis_name}, Axis Content: {axis}, Type: {type(axis)}, Shape: {getattr(axis, 'shape', 'No Shape')}, Length: {len(axis) if hasattr(axis, '__len__') else 'No Length'}")
                    #         return [
                    #             PoseTimeseries.plot_pivot_axes(axis, axis_name),
                    #             PoseTimeseries.plot_pivot_axes_fft(axis, axis_name) if len(axis.shape) == 1 and axis.shape[0] > 0 else None
                    #         ]
                    #     vm_axis.change(_on_axis_change, inputs=[vm_axis], outputs=[plot, fft_plot])
                        
                    #     def _on_is_filtered_change(x_is_filtered, pivot, filter_fn=lambda x: x, filter_params=[]):
                    #         if not filter_fn: return []
                    #         fn = lambda x: filter_fn(x, *filter_params)
                    #         return PoseTimeseries.filter_axis(pivot, fn)
                    #     gr.on(
                    #         triggers=[is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change], 
                    #         fn=_on_is_filtered_change,
                    #         inputs=[is_filtered, vm.pivot[pi], vm.filter, vm.filter_params],
                    #         outputs=[vm_axis]
                    #     )
        
                    gr.Markdown("#### X Axis")
                    x_is_filtered = gr.Checkbox(label="Use Filter (X)")

                    x_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    x_fft_plot = gr.Image()
                    def _on_x_axis_change(x_axis):
                        return [
                            PoseTimeseries.plot_pivot_axes(x_axis, "X"),
                            PoseTimeseries.plot_pivot_axes_fft(x_axis, "X") if len(x_axis) > 0 else None
                        ]
                    vm.x_axis[pi].change(_on_x_axis_change, inputs=[vm.x_axis[pi]], outputs=[x_plot, x_fft_plot])
                    

                    gr.Markdown("#### Y Axis")
                    y_is_filtered = gr.Checkbox(label="Use Filter (Y)")

                    y_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    y_fft_plot = gr.Image()
                    def _on_y_axis_change(y_axis):
                        return [
                            PoseTimeseries.plot_pivot_axes(y_axis, "Y"),
                            PoseTimeseries.plot_pivot_axes_fft(y_axis, "Y") if len(y_axis) > 0 else None
                        ]
                    vm.y_axis[pi].change(_on_y_axis_change, inputs=[vm.y_axis[pi]], outputs=[y_plot, y_fft_plot])

                    gr.Markdown("#### Z Axis")
                    z_is_filtered = gr.Checkbox(label="Use Filter (Z)")

                    z_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    z_fft_plot = gr.Image()
                    def _on_z_axis_change(z_axis):
                        return [
                            PoseTimeseries.plot_pivot_axes(z_axis, "Z"),
                            PoseTimeseries.plot_pivot_axes_fft(z_axis, "Z") if len(z_axis) > 0 else None
                        ]
                    vm.z_axis[pi].change(_on_z_axis_change, inputs=[vm.z_axis[pi]], outputs=[z_plot, z_fft_plot])                        

                    def _on_is_filtered_change_factory(axis):
                        def _on_is_filtered_change(is_filtered, pivot, filter_fn=lambda x: x, filter_params=[]):
                            if not filter_fn: return pivot[axis]
                            fn = lambda x: filter_fn(x, *filter_params)
                            return PoseTimeseries.filter_axis(pivot[axis], fn) if is_filtered else pivot[axis]
                        return _on_is_filtered_change
                    
                    gr.on(
                        triggers=[x_is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change], 
                        fn=_on_is_filtered_change_factory(0),
                        inputs=[x_is_filtered, vm.pivot[pi], vm.filter, vm.filter_params],
                        outputs=[vm.x_axis[pi]]
                    )
                    
                    gr.on(
                        triggers=[y_is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change], 
                        fn=_on_is_filtered_change_factory(1),
                        inputs=[y_is_filtered, vm.pivot[pi], vm.filter, vm.filter_params],
                        outputs=[vm.y_axis[pi]]
                    )
                    
                    gr.on(
                        triggers=[z_is_filtered.change, vm.pivot[pi].change, vm.filter.change, vm.filter_params.change], 
                        fn=_on_is_filtered_change_factory(2),
                        inputs=[z_is_filtered, vm.pivot[pi], vm.filter, vm.filter_params],
                        outputs=[vm.z_axis[pi]]
                    )
                        
                    
                    gr.Markdown("## XY 2D Vector magnitude")
                    v2_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    v2_fft_plot = gr.Image()
                    gr.Markdown("## XYZ 3D Vector magnitude")
                    v3_plot = gr.Image()
                    gr.Markdown("###### FFT")
                    v3_fft_plot = gr.Image()
                    
                    def update_v2_plot(x_axis, y_axis):
                        pivot = np.array([x_axis, y_axis, np.zeros_like(x_axis)])
                        mock_pose_timeseries = np.array([pivot])
                        vector2d = Vector2dTimeseries.pose_to_vector_length(mock_pose_timeseries)[0]
                        return [
                            Vector2dTimeseries.plot_pivot(vector2d),
                            Vector2dTimeseries.plot_pivot_fft(vector2d, "vector XY magnitude") if len(vector2d) > 0 else None
                        ]
                    gr.on(
                        triggers=[vm.x_axis[pi].change, vm.y_axis[pi].change], 
                        fn=update_v2_plot,
                        inputs=[vm.x_axis[pi], vm.y_axis[pi]], 
                        outputs=[v2_plot, v2_fft_plot]
                    )
                    
                    def update_v3_plot(x_axis, y_axis, z_axis):
                        pivot = np.array([x_axis, y_axis, z_axis])
                        mock_pose_timeseries = np.array([pivot])
                        vector3d = Vector3dTimeseries.pose_to_vector_length(mock_pose_timeseries)[0]
                        return [
                            Vector3dTimeseries.plot_pivot(vector3d, rank="3D"),
                            Vector3dTimeseries.plot_pivot_fft(vector3d, "vector XYZ magnitude") if len(vector3d) > 0 else None
                        ]
                    gr.on(
                        triggers=[vm.x_axis[pi].change, vm.y_axis[pi].change, vm.z_axis[pi].change], 
                        fn=update_v3_plot,
                        inputs=[vm.x_axis[pi], vm.y_axis[pi], vm.z_axis[pi]], 
                        outputs=[v3_plot, v3_fft_plot]
                    )
                
        
            for part_index in range(2):
                part_config(part_index)
                
                
        self.file_selector = file_selector