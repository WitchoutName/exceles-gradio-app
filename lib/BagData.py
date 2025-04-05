import os
import numpy as np
from PIL import Image
import cv2
import gradio as gr
from tqdm import tqdm
from lib.PoseTimeseries import PoseTimeseries, body_part_names
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from config import DEBUG


class BagData:
    COMPUTED_DIR = "0_computed"
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bag_name = os.path.basename(data_dir).split(".")[0]
        self.video_path = os.path.join(data_dir, BagData.COMPUTED_DIR, "video.mp4")
        self.frames = []
        # Processing
        self.pose3d_array = []
        self.pose_timeseries = PoseTimeseries.load(self.data_dir)
        self.duration = self.get_frame_count() / 60

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
    
    def get_frame_count(self):
        if self.pose_timeseries is not None:
            return self.pose_timeseries[list(self.pose_timeseries.keys())[0]].shape[2]
        return len([f for f in os.listdir(self.data_dir) if f.startswith("frame_")])
    
    def get_video(self, cache=True):
        if cache and os.path.exists(self.video_path):
            return self.video_path
    
        def processed_frame(index):
            frame_path = os.path.join(self.data_dir, f"frame_{str(index).zfill(5)}")
            frame_image = self.load_frame(frame_path, color=True, depth=False, confidence=False)["color"]
            
            def get_pose_coords(pose):
                return [pose[0][index], pose[1][index]]
            
            body = self.pose_timeseries["body"]
            neck = body[body_part_names.index("Neck")]
            eye_left = body[body_part_names.index("Left Eye")]
            eye_right = body[body_part_names.index("Right Eye")]
            coords = [get_pose_coords(eye_left), *[get_pose_coords(eye_right), get_pose_coords(neck)]*2]
            face_center = tuple(map(int, np.mean(coords, axis=0)))
            
            # Draw circle on face center
            radius = int(np.linalg.norm(np.array(coords[0]) - np.array(face_center)))
            frame_image = cv2.circle(np.array(frame_image), face_center, int(radius * 1.2), (0, 0, 0), -1)
            
            return frame_image

        frame_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("frame_")])
        images = [processed_frame(index) for index in tqdm(range(len(frame_files)-1), desc="Creating video")]

        clip = ImageSequenceClip(images, fps=60)
        clip.write_videofile(self.video_path)
        
        return self.video_path
        
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
    