from lib.PoseTimeseries import PoseTimeseries
import numpy as np
import matplotlib.pyplot as plt



class Vector2dTimeseries(PoseTimeseries):
    FILE_NAME_PREFIX = "vector2d_timeseries"
    LABEL = "Vector 2D size"

    @staticmethod
    def pose_to_vector_length(body_part):
        """
        Computes the 2D vector length (ignoring the z-dimension) over time for each pivot in a body part.
        Parameters:
            body_part: np.array of shape (n, 3, t_length)
                n: number of pivots (e.g., joints),
                3: x, y, z coordinates,
                t_length: number of time frames.
        Returns:
            np.array of shape (n, t_length): The computed 2D vector lengths over time for each pivot.
        """
        # Extract x, y components and compute Euclidean norm along time dimension
        vector_lengths = np.linalg.norm(body_part[:, :2, :], axis=1)
        return vector_lengths
        
    @classmethod
    def from_pose_timeseries(cls, pose_timeseries):
        return {
            "body": cls.pose_to_vector_length(pose_timeseries["body"]),
            "hand_left": cls.pose_to_vector_length(pose_timeseries["hand_left"]),
            "hand_right": cls.pose_to_vector_length(pose_timeseries["hand_right"])
        }

    @classmethod
    def plot_fourier_transform(cls, data, fps=60):
        """
        Applies Fourier Transform to the array and plots the.
        Parameters:
            data (numpy.ndarray): Input array of shape (n)
            fps (int): Frame rate of the footage in frames per second
        """
        # Apply Fourier Transform
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
        plt.figure(figsize=(14, 5))
        plt.plot(freq, magnitude, label=cls.LABEL)
        plt.yscale('log')
        plt.title(f'Fourier Transform of {cls.LABEL}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        # plt.xlim(0, 10) # Limit x-axis to 10 kHz if needed
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pivot(data):
        """
        Plots the array over time.
        Parameters:
        data (numpy.ndarray): Input array of shape (n).
        """
        # Create a figure with 3 subplots
        # Plot each coordinate
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title('Vector size Over Time')
        plt.xlabel('Time')
        plt.ylabel('Vector size')
        plt.legend()
        plt.tight_layout()
        plt.show()