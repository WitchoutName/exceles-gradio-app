from lib.Vector2dTimeseries import Vector2dTimeseries
import numpy as np
import matplotlib.pyplot as plt


class Vector3dTimeseries(Vector2dTimeseries):
    FILE_NAME_PREFIX = "vector3d_timeseries"
    LABEL = "Vector 3D size"
    
    @staticmethod
    def pose_to_vector_length(body_part):
        """
        Computes the 3D vector length (including x, y, z dimensions) over time for each pivot in a body part.
    
        Parameters:
            body_part: np.array of shape (n, 3, t_length)
                n: number of pivots (e.g., joints),
                3: x, y, z coordinates,
                t_length: number of time frames.
    
        Returns:
            np.array of shape (n, t_length): The computed 3D vector lengths over time for each pivot.
        """
        # Compute Euclidean norm across the x, y, and z dimensions over time
        vector_lengths = np.linalg.norm(body_part, axis=1)
        return vector_lengths