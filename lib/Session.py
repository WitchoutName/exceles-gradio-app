import uuid
import gradio as gr

class Session:
    def __init__(self, app: gr.Blocks, on_ready=lambda: None):
        self.app = app
        self.session_hash = None
        self.state_key_dict = {}
        self.on_ready = on_ready
        
    def update(self):
        self.state_key_dict = {v: k for k, v in self.app.blocks.items() if isinstance(v, gr.State)}
        
    def inject_tag(self):
        tag = gr.State("tag")
        @tag.change(inputs=[tag])
        def _on_tag_change(tag):
            self.session_hash = next((hash for hash, data in app.state_holder.session_data.items() if next(iter(data.state_data.values()), None) == tag), None)
            self.update()
            self.on_ready()
        
        app = self.app        
        @app.load(outputs=[tag])
        def _on_load():
            return str(uuid.uuid4())
        
    
    def get_value(self, state_obj):
        key = self.state_key_dict.get(state_obj)
        if key is None:
            return None
        return self.app.state_holder.session_data.get(self.session_hash).state_data.get(key)