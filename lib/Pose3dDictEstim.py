import numpy as np
import cv2
from tqdm import tqdm


from lib.Pose3dDict import Pose3dDict


class Pose3dDictEstim(Pose3dDict):
    FILE_NAME = "pose3d_dict_estim.npz"

    @staticmethod
    def approximate_depth(depth_image, x, y, percentage=0.2, max_radius=100):
        """
        If the depth on given coordinates is null (0), returns the average of the surrounding circle area.
        Radius is increased until at least 20% of the pixels in the radius have non-null values.
        depth_image: PIL image
        x: x coordinate
        y: y coordinate
        @returns: depth (float), redius (int)
        """
        depth_array = np.array(depth_image)
        height, width = depth_array.shape
        y, x = np.clip(int(y), 0, height-1), np.clip(int(y), 0, width-1)
    
        if depth_array[y, x] != 0:
            return depth_array[y, x], 0
    
        radius = 1
        while True:
            non_null_values = []
            for i in range(max(0, y - radius), min(height, y + radius + 1)):
                for j in range(max(0, x - radius), min(width, x + radius + 1)):
                    if (i - y) ** 2 + (j - x) ** 2 <= radius ** 2 and depth_array[i, j] != 0:
                        non_null_values.append(depth_array[i, j])
    
            if len(non_null_values) >= percentage * (np.pi * radius ** 2):
                return np.mean(non_null_values), radius
            elif radius >= max_radius:
                return -1, radius
    
            radius += 1

    @classmethod
    # Convert keypoints to 3D coordinates using depth interpolation
    def keypoints_to_3d(cls, keypoints: list, frame, confidence_threashold = 0.5) -> np.ndarray:
        keypoints_3d = []
        depth_frame = frame["depth"]
        
        for i in range(0, len(keypoints), 3):
            x, y, _ = keypoints[i:i+3]
            y_shape, x_shape = depth_frame.shape
            depth = depth_frame[np.clip(int(y), 0, y_shape-1), np.clip(int(y), 0, x_shape-1)]
            depth, radius = cls.approximate_depth(depth_frame, int(x), int(y))
            keypoints_3d.append(x)
            keypoints_3d.append(y)
            keypoints_3d.append(depth)
        
        return np.array(keypoints_3d)


def transform_pose3d_array(pose3d_array):
    def reshape_keypoints(keypoints_list):
        keypoints_np = np.array(keypoints_list)
        num_frames = keypoints_np.shape[0]
        num_keypoints = keypoints_np.shape[1] // 3
        keypoints_np = keypoints_np.reshape(num_frames, num_keypoints, 3)
        return keypoints_np.transpose(1, 2, 0)  # Shape: (num_keypoints, 3, num_frames)

    timeseries = {
        "body": reshape_keypoints([frame["body_keypoints_3d"] for frame in pose3d_array]),
        "hand_left": reshape_keypoints([frame["hand_left_keypoints_3d"] for frame in pose3d_array]),
        "hand_right": reshape_keypoints([frame["hand_right_keypoints_3d"] for frame in pose3d_array]),
    }

    return timeseries