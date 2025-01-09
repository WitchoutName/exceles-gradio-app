import gradio as gr

from views.ParseView import ParseView
from views.ProcessView import ProcessView
from views.VisualizeView import VisualizeView

with gr.Block() as app:
    with gr.TabGroup() as tabs:
        with gr.Tab("Parse"):
            parse_view = ParseView()
        with gr.Tab("Detect pose"):
            process_view = ProcessView()
        with gr.Tab("Visualize"):
            visu_view = VisualizeView()
            
            
app.launch()