import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import DwposeDetector
import traceback
from lib.BagData import BagData

pose_detector = None


def process_frame(data_path, frame_name, process_method, use_cache=True):
    try:
        frame = BagData.load_frame(os.path.join(data_path, frame_name))
        retult = process_method(frame, use_cache)
        return retult
    except Exception as e:
        traceback.print_exc()
        raise e




class Pose3dDict:
    FILE_NAME = "pose3d_dict.npz"
    
    @classmethod
    def save(cls, frame_path, pose3d_dict):
        print(f"saving cache {os.path.join(frame_path, cls.FILE_NAME)}")
        np.savez(os.path.join(frame_path, cls.FILE_NAME), 
                 body_keypoints_3d=pose3d_dict["body_keypoints_3d"],
                 hand_left_keypoints_3d=pose3d_dict["hand_left_keypoints_3d"],
                 hand_right_keypoints_3d=pose3d_dict["hand_right_keypoints_3d"])

    @classmethod
    def load(cls, frame_path):
        print(f"loading cache {os.path.join(frame_path, cls.FILE_NAME)}")
        data = np.load(os.path.join(frame_path, cls.FILE_NAME))
        return {
            "body_keypoints_3d": data["body_keypoints_3d"],
            "hand_left_keypoints_3d": data["hand_left_keypoints_3d"],
            "hand_right_keypoints_3d": data["hand_right_keypoints_3d"]
        }

    @classmethod
    def from_unpacked_bag(cls, data_path, use_cache=True):
        global pose_detector
        pose_detector = DwposeDetector.from_pretrained(
            pretrained_model_or_path="yzd-v/DWPose",
            pretrained_det_model_or_path="yzd-v/DWPose",
        )
        pose3d_array = []
        

        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_frame, data_path, frame_name, cls.get, use_cache): frame_name for frame_name in sorted(os.listdir(data_path))[2:]}
            error_occurred = False
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
                if error_occurred:
                    break
                try:
                    pose3d_array.append(future.result())
                    
                except Exception as e:
                    error_occurred = True
                    executor.shutdown(wait=False)
                    traceback.print_exc()
                    return f"Error processing frame {futures[future]}: {e}", None
        return None, pose3d_array

    @classmethod
    def get(cls, frame, use_cache=True):
        if cls.FILE_NAME in os.listdir(frame["path"]) and use_cache:
            return cls.load(frame["path"])
        
        detected_image, pose_dict = pose_detector(
            input_image=frame["color"],
            #detect_resolution=256,
            include_body=True,
            include_hand=True,
            include_face=False,
            output_type="pil",
            image_and_json=True
        )
    
        person_pose = pose_dict.get("people")[0]
        body_points = person_pose.get("pose_keypoints_2d") or [0]*18*3
        handl_points = person_pose.get("hand_left_keypoints_2d") or [0]*21*3
        handr_points = person_pose.get("hand_right_keypoints_2d") or [0]*21*3
        pose3d_dict = {
            "body_keypoints_3d": cls.keypoints_to_3d(body_points, frame),
            "hand_left_keypoints_3d": cls.keypoints_to_3d(handl_points, frame),
            "hand_right_keypoints_3d": cls.keypoints_to_3d(handr_points, frame),
        }
        
        cls.save(frame["path"], pose3d_dict)
        return pose3d_dict

    @classmethod
    def remove_all(cls, data_path):
        for frame in sorted(os.listdir(data_path))[1:]:
            condition = int(frame.split("_")[1]) > 198
            condition = True
            if condition:
                try:
                    os.remove(f"{data_path}/{frame}/{cls.FILE_NAME}")
                except:
                    pass
    
    @staticmethod
    # Convert keypoints to 3D coordinates using depth based on confidence matrix
    def keypoints_to_3d(keypoints: list, frame, confidence_threashold = 0.5) -> np.ndarray:
        keypoints_3d = []
        confidence_min = np.min(frame["confidence"])
        confidence_max = np.max(frame["confidence"])
        confidence_frame = (frame["confidence"] - confidence_min) / (confidence_max - confidence_min)
        depth_frame = frame["depth"]
        
        for i in range(0, len(keypoints), 3):
            x, y, _ = keypoints[i:i+3]
            confidence = confidence_frame[int(y), int(x)]
            depth = depth_frame[int(y), int(x)]
            keypoints_3d.append(x)
            keypoints_3d.append(y)
            keypoints_3d.append(depth if confidence > confidence_threashold else -1)
        
        return np.array(keypoints_3d)
    
