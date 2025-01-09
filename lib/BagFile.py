import os
import traceback
import shutil
import pyrealsense2 as rs
import numpy as np
import cv2
from tqdm import tqdm

from lib.BagData import BagData


def skip_frames(pipeline, num_frames):
    print(f"Skipping {num_frames} frames...")
    for _ in tqdm(range(num_frames)):
        pipeline.wait_for_frames()


def save_frames(frames, frame_number, output_dir):
    # Extract frames
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    confidence_frame = frames.first_or_default(rs.stream.confidence)

    if not color_frame or not depth_frame or not confidence_frame:
        return False

    # Convert frames to NumPy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    confidence_image = np.asanyarray(confidence_frame.get_data())

    # Normalize depth and confidence for visualization (optional)
    depth_norm = (depth_image / depth_image.max() * 255).astype(np.uint8)
    confidence_norm = (confidence_image / confidence_image.max() * 255).astype(np.uint8)

    # Save data
    frame_dir = os.path.join(output_dir, f"frame_{frame_number:05d}")
    os.makedirs(frame_dir, exist_ok=True)

    # Save RGB image as PNG
    color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(frame_dir, "image.png"), color_image_bgr)
    cv2.imwrite(os.path.join(frame_dir, "image.png"), color_image_bgr)

    # Save depth and confidence as NumPy arrays
    np.save(os.path.join(frame_dir, "depth.npy"), depth_image)
    np.save(os.path.join(frame_dir, "confidence.npy"), confidence_image)


    # Optionally save normalized visualizations
    # cv2.imwrite(os.path.join(frame_dir, "depth_visual.png"), depth_norm)
    # cv2.imwrite(os.path.join(frame_dir, "confidence_visual.png"), confidence_norm)
    return True


def extract_data_from_bag(bag_file, output_dir, LOWER_LIMIT=500, UPPER_LIMIT=1500, delete_previous=False):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream from the .bag file
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    # Start streaming from file
    pipeline_profile = pipeline.start(config)

    # Get the playback device
    playback = pipeline_profile.get_device().as_playback()
    playback.set_real_time(False)

    if delete_previous:
        if os.path.exists(output_dir):
            os.system(f"rm -r {output_dir}")
    os.makedirs(output_dir, exist_ok=True)


    frame_number = max(LOWER_LIMIT, 0)
    try:
        skip_frames(pipeline, LOWER_LIMIT)
        with tqdm(total=UPPER_LIMIT-LOWER_LIMIT, desc="Parsing frames") as pbar:
            while  True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                if not save_frames(frames, frame_number, output_dir):
                    break
                print(frame_number, end="\r")
                frame_number += 1
                pbar.update(1)

    except:
        traceback.print_exc()
    finally:
        pipeline.stop()


def get_bag_frame_count(bag_file, use_tqdm=True, standard_frame_count=8000):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream from the .bag file
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)

    # Start streaming from file
    pipeline_profile = pipeline.start(config)

    # Get the playback device
    playback = pipeline_profile.get_device().as_playback()
    playback.set_real_time(False)

    frame_count = 0
    try:
        if use_tqdm:
            progress = tqdm(total=standard_frame_count, desc="Counting frames")
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            frame_count += 1
            if use_tqdm:
                progress.update(1)
            print(frame_count, depth_frame, end="\r")
            if not depth_frame:
                break
    except Exception as e:
        print(e)

    # Get the frame count
    duration = playback.get_duration()
    framerate = frame_count / duration.total_seconds()
    
    # Stop the pipeline
    pipeline.stop()

    return frame_count, duration, framerate


class BagFile:
    @staticmethod
    def list_raw_files(base_dir):
        return [file for file in os.listdir(base_dir) if file.endswith(".bag")]
    
    @staticmethod
    def upload_file(temp_path, base_dir):
        try:
            file_name = os.path.basename(temp_path)
            if not file_name.endswith(".bag"):
                return [None, f"File {file_name} is not a .bag file"]
            target_path = os.path.join(base_dir, file_name)
            shutil.move(temp_path, target_path)
            return [file_name, f"File {file_name} uploaded successfully"]
        except Exception as e:
            traceback.print_exc()
            return [None, f"Error uploading file: {str(e)}"]
        
    @staticmethod
    def get_stats(file_path):        
        frame_count, duration, framerate = get_bag_frame_count(file_path)
        return frame_count, duration.total_seconds(), framerate
        
    @staticmethod
    def parse(file_path, output_folder, count=True):
        try:
            if count:
                frame_count, duration, framerate = get_bag_frame_count(file_path, False)
                computed_dir = os.path.join(output_folder, BagData.COMPUTED_DIR)
                os.makedirs(computed_dir, exist_ok=True)
                with open(os.path.join(computed_dir, "stats.txt"), "w") as f:
                    f.write(f"Frame count: {frame_count}\n")
                    f.write(f"Duration: {duration}\n")
                    f.write(f"Framerate: {framerate}\n")
                extract_data_from_bag(file_path, output_folder, 0, frame_count, delete_previous=True)
            else:        
                extract_data_from_bag(file_path, output_folder, -1, -1, delete_previous=True)
                return [True, f"File {file_path} parsed successfully"]
        except Exception as e:
            traceback.print_exc()
            return [False, f"Error parsing file: {str(e)}"] 
        
         