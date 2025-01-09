import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter



class Filter:
    @staticmethod
    def moving_average(data, window_size_percentage=0.05):
        window_size = int(data.shape[0] * window_size_percentage)
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    @staticmethod
    def gaussian(data, sigma=2):
        return gaussian_filter(data, sigma=sigma)

    @staticmethod
    def savgol(data, window_length=11, polyorder=2):
        return savgol_filter(data, window_length=window_length, polyorder=polyorder)
