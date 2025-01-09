from typing import List, Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

from config import DEBUG

# Define body and hand part names
body_part_names = [
    "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
    "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip",
    "Right Knee", "Right Ankle", "Left Hip", "Left Knee",
    "Left Ankle", "Right Eye", "Left Eye", "Right Ear", "Left Ear"
]

hand_part_names = [
    "Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
    "Index Finger MCP", "Index Finger PIP", "Index Finger DIP",
    "Index Finger Tip", "Middle Finger MCP", "Middle Finger PIP",
    "Middle Finger DIP", "Middle Finger Tip", "Ring Finger MCP",
    "Ring Finger PIP", "Ring Finger DIP", "Ring Finger Tip",
    "Pinky Finger MCP", "Pinky Finger PIP", "Pinky Finger DIP",
    "Pinky Finger Tip"
]

class PoseTimeseries:
    FILE_NAME_PREFIX = "pose_timeseries"
    DIR_NAME = "0_computed"
    all_parts_list = body_part_names + ["Left "+x for x in hand_part_names] + ["Right "+x for x in hand_part_names]

    
    @classmethod
    def get(cls, pose3d_array):
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
    
    @classmethod
    def save(cls, data_path, pose_timeseries, min_index = None, max_index = None, suffix=""):
        data_path = os.path.join(data_path, cls.DIR_NAME)
        os.makedirs(data_path, exist_ok=True)
        separator = "<>" if min_index or max_index else ""
        index_string = f"{min_index if min_index else ''}{separator}{max_index if max_index else ''}"
        suffix += ".npz" if DEBUG else "__prod.npz"
        file_path = os.path.join(data_path, f"{cls.FILE_NAME_PREFIX}{index_string}{suffix}")
        np.savez(file_path, 
                 body=pose_timeseries["body"],
                 hand_left=pose_timeseries["hand_left"],
                 hand_right=pose_timeseries["hand_right"])
    
    @staticmethod
    def load_file(file_path):
        data = np.load(file_path)
        return {
            "body": data["body"],
            "hand_left": data["hand_left"],
            "hand_right": data["hand_right"]
        }

    @classmethod
    def load(cls, data_path, index=-1):
        data_path = os.path.join(data_path, cls.DIR_NAME)
        if index < 0:
            # TODO: ensure that the file exists
            file_name = list(filter(lambda x: cls.FILE_NAME_PREFIX in x and x.endswith(".npz" if DEBUG else "__prod.npz"), os.listdir(data_path)))[0]
            return cls.load_file(os.path.join(data_path, file_name))
        os.makedirs(data_path, exist_ok=True)
        files = [file for file in os.listdir(data_path) if cls.FILE_NAME_PREFIX in file]
        return cls.load_file(os.path.join(data_path, files[index]))

    @classmethod
    def list(cls, data_path):
        data_path = os.path.join(data_path, cls.DIR_NAME)
        os.makedirs(data_path, exist_ok=True)
        files = [file for file in os.listdir(data_path) if cls.FILE_NAME_PREFIX in file]
        for i, file in enumerate(files):
            print(f"[{i}] {file}")

    @staticmethod
    def filter_axis(data, filter_function):
        original_length = data.shape[0]
        filtered_data = filter_function(data)
        
        def ensure_length(signal, length):
            if len(signal) < length:
                # Pad with np.nan (or 0 if preferred) to match the original length
                return np.pad(signal, (0, length - len(signal)), constant_values=np.nan)
            elif len(signal) > length:
                # Truncate to the original length
                return signal[:length]
            return signal
        
        return ensure_length(filtered_data, original_length)


    @staticmethod
    def filter_pivot(pivot, filter_function, **kwargs):
        """
        Parameters:
            pivot (np.array of shape (3, t_length))
            filter_function (function)
            kwargs:
                x (bool): whether to filter the x axis (default False)
                y (bool): whether to filter the y axis (default False)
                z (bool): whether to filter the z axis (default True)
        """
        x = kwargs.get("x", False)
        y = kwargs.get("y", False)
        z = kwargs.get("z", True)
        original_length = pivot.shape[1]  # This is t_length
        
        filtered_x = pivot[0] if not x else filter_function(pivot[0])
        filtered_y = pivot[1] if not y else filter_function(pivot[1])
        filtered_z = pivot[2] if not z else filter_function(pivot[2])
    
        # Ensure all filtered signals match the original length by padding or truncating
        def ensure_length(signal, length):
            if len(signal) < length:
                # Pad with np.nan (or 0 if preferred) to match the original length
                return np.pad(signal, (0, length - len(signal)), constant_values=np.nan)
            elif len(signal) > length:
                # Truncate to the original length
                return signal[:length]
            return signal
    
        return np.array([
            ensure_length(filtered_x, original_length),
            ensure_length(filtered_y, original_length),
            ensure_length(filtered_z, original_length)
        ])


    @classmethod
    def filter(cls, data, filter_function):
        return {
            key: np.array([cls.filter_pivot(pivot, filter_function) for pivot in data[key]])
            for key in data
        }
        
    @classmethod
    def get_body_part_object_path(cls, body_part_name):
        """
        Returns the object path of the body part (pose_dict_key: Str, index: int).
        """
        index = cls.all_parts_list.index(body_part_name)
        if index < len(body_part_names):
            return "body", body_part_names.index(body_part_name)
        elif index < len(body_part_names) + len(hand_part_names):
            return "hand_left", hand_part_names.index(body_part_name[len("Left "):])
        else:
            return "hand_right", hand_part_names.index(body_part_name[len("Right "):])
            

    @staticmethod
    def plot_pivot_axes(data, axis_label):
        """
        Plots the values of a single axis over time.
        Parameters:
            data (numpy.ndarray): Input array of shape (n,)
            axis_label (str): Label for the axis
        Returns:
            PIL.Image: Image of the plot       
        """
        if data is None: return None
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.plot(data, label=f'{axis_label} Component')
        ax.set_title(f'{axis_label} Values Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{axis_label} Values')
        ax.grid(True)
        
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)
            
    @staticmethod
    def plot_pivot(data):
        """
        Plots the x, y, z coordinates over time.
        
        Parameters:
        data (numpy.ndarray): Input array of shape (3, n) where each row corresponds to x, y, z coordinates.
        """
        # Check if input data has the correct shape
        if data.shape[0] != 3:
            raise ValueError("Input data must have shape (3, n)")
    
        # Create a time array based on the number of samples
        n = data.shape[1]
        time = np.arange(n)
    
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
        # Plot each coordinate
        axs[0].plot(time, data[0], label='X Coordinate', color='r')
        axs[0].set_title('X Coordinate Over Time')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('X Coordinate')
        axs[0].grid()
        axs[0].legend()
    
        axs[1].plot(time, data[1], label='Y Coordinate', color='g')
        axs[1].set_title('Y Coordinate Over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Y Coordinate')
        axs[1].grid()
        axs[1].legend()
    
        axs[2].plot(time, data[2], label='Z Coordinate', color='b')
        axs[2].set_title('Z Coordinate Over Time')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Z Coordinate')
        axs[2].grid()
        axs[2].legend()
    
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_pivot_axes_fft(data, axis_label, fps=60):
        """
        Plots the values of a single axis over time.
        Parameters:
            data (numpy.ndarray): Input array of shape (n,)
            axis_label (str): Label for the axis
        Returns:
            PIL.Image: Image of the plot       
        """
        if data is None: return None
        fig, ax = plt.subplots(figsize=(14, 6))
        
        fourier = np.fft.fft(data)
        # Get the number of samples
        n = data.shape[0]
        # Calculate frequency bins in Hz
        freq = np.fft.fftfreq(n, d=1/fps)
        # Only take the positive half of the frequencies and corresponding magnitudes
        half_n = n // 2
        freq = freq[:half_n]
        magnitude = np.abs(fourier[:half_n])
        # Plotting the magnitude of the Fourier Transform
        ax.plot(freq, magnitude, label=f'{axis_label}', color="g")
        ax.set_yscale('log')
        ax.set_title(f'Fourier Transform of {axis_label}')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        # axs[i].set_xlim(0, 10) # Limit x-axis to 10 kHz
        ax.grid(True)
        
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

    @staticmethod
    def plot_fourier_transform(data, fps=60):
        """
        Applies Fourier Transform to each of the three subarrays in a (3, n) array
        and plots the results side by side.
        Parameters:
            data (numpy.ndarray): Input array of shape (3, n)
            fps (int): Frame rate of the footage in frames per second
        """
        # Check if input data has the correct shape
        if data.shape[0] != 3:
            raise ValueError("Input data must have shape (3, n)")
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # Loop through each subarray
        for i, point in enumerate(["x", "y", "z"]):
            # Apply Fourier Transform
            fourier = np.fft.fft(data[i])
            # Get the number of samples
            n = data.shape[1]
            # Calculate frequency bins in Hz
            freq = np.fft.fftfreq(n, d=1/fps)
            # Only take the positive half of the frequencies and corresponding magnitudes
            half_n = n // 2
            freq = freq[:half_n]
            magnitude = np.abs(fourier[:half_n])
            # Plotting the magnitude of the Fourier Transform
            axs[i].plot(freq, magnitude, label=f'{point}')
            axs[i].set_yscale('log')
            axs[i].set_title(f'Fourier Transform of {point}')
            axs[i].set_xlabel('Frequency (Hz)')
            axs[i].set_ylabel('Magnitude')
            # axs[i].set_xlim(0, 10) # Limit x-axis to 10 kHz
            axs[i].grid()
            axs[i].legend()
        plt.tight_layout()
        plt.show()


    @classmethod
    def plot_fft(cls, pose_timeseries):
        def plot_bodypart(part_names, part_name, side):
            part = pose_timeseries[part_name]
            for i in range(part.shape[0]):
                print(side, part_names[i])
                cls.plot_fourier_transform(part[i])
        def plot_hand(side):
            hand_name = f"hand_{side.lower()}"
            plot_bodypart(hand_part_names, hand_name, side)
        def plot_body():
            plot_bodypart(body_part_names, "body", "")
    
        print("================== BODY =================")
        plot_body()
        print("=============== LEFT HAND ===============")
        plot_hand("Left")
        print("=============== RIGHT HAND ===============")
        plot_hand("Right")