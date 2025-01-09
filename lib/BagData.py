import os
import numpy as np
from PIL import Image
import cv2
import gradio as gr
from tqdm import tqdm
from lib.PoseTimeseries import PoseTimeseries
import moviepy.video.io.ImageSequenceClip

from config import DEBUG


class BagData:
    COMPUTED_DIR = "0_computed"
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bag_name = os.path.basename(data_dir).split(".")[0]
        self.video_path = os.path.join(data_dir, BagData.COMPUTED_DIR, "video.mp4")
        self.frame_count = len(os.listdir(data_dir)) - 1
        self.duration = self.frame_count / 60
        self.frames = []
        # Processing
        self.pose3d_array = []
        self.pose_timeseries = PoseTimeseries.load(self.data_dir)

    @staticmethod
    def load_frame(frame_path, **kwargs):
        color = Image.open(os.path.join(frame_path, "image.png")).convert("RGB") if kwargs.get("color") is not False else None
        depth = np.load(os.path.join(frame_path, "depth.npy")) if kwargs.get("depth") is not False else None
        confidence = np.load(os.path.join(frame_path, "confidence.npy")) if kwargs.get("confidence") is not False else None
        return {
            "color": color,
            "depth": cv2.resize(depth, (color.width, color.height), interpolation=cv2.INTER_NEAREST) if depth is not None else None,
            "confidence": cv2.resize(confidence, (color.width, color.height), interpolation=cv2.INTER_NEAREST) if confidence is not None else None,
            "path": frame_path
        }
    
    @staticmethod
    def list_parsed(base_dir):
        res = []
        for bag_data in os.listdir(base_dir):
            data_path = os.path.join(base_dir, bag_data)
            if not os.path.isdir(data_path): continue
            data_path_content = sorted(os.listdir(data_path))
            if data_path.endswith(".bag_data") and len(data_path_content) > 0:
                for file in data_path_content:
                    if file.startswith("frame_"):
                        res.append(bag_data)
                        break
        return res

        
    @staticmethod
    def list_processed(base_dir):
        res = []
        for bag_data in os.listdir(base_dir):
            data_path = os.path.join(base_dir, bag_data)
            if not os.path.isdir(data_path): continue
            data_path_content = sorted(os.listdir(data_path))
            if data_path.endswith(".bag_data") and len(data_path_content) > 0 and data_path_content[0] == BagData.COMPUTED_DIR:
                for file in os.listdir(os.path.join(data_path, BagData.COMPUTED_DIR)):
                    if file.startswith("pose_timeseries") and file.endswith(".npz" if DEBUG else "__prod.npz"):
                        res.append(bag_data.split(".")[0])
                        break
        print(res)
        return res
    
    def get_video(self, cache=True):
        if cache and os.path.exists(self.video_path):
            return self.video_path
    
        first_frame = self.load_frame(os.path.join(self.data_dir, sorted(os.listdir(self.data_dir))[2]), color=True)["color"]    

        image_files = [os.path.join(self.data_dir, frame, "image.png") for frame in sorted(os.listdir(self.data_dir))[1:]]

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=60)
        clip.write_videofile(self.video_path)
        
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        # video_writer = cv2.VideoWriter(self.video_path, fourcc, 60, first_frame.size)
        
        # for frame in tqdm(sorted(os.listdir(self.data_dir))[1:], leave=False):
        #     frame_path = os.path.join(self.data_dir, frame)
        #     color_frame = self.load_frame(frame_path)["color"]
        #     color_frame_np = np.array(color_frame)
            
        #     # Ensure the frame size is consistent
        #     if color_frame_np.shape[:2] != first_frame.size:
        #         color_frame_np = cv2.resize(color_frame_np, first_frame.size, interpolation=cv2.INTER_AREA)
            
        #     video_writer.write(color_frame_np)
        #     # video.write(cv2.cvtColor(color_frame_np, cv2.COLOR_RGB2BGR))
        
        # video_writer.release()
        return self.video_path
    