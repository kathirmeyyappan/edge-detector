"""
This file is meant to aid in utilizing the theta value from the Canny edge 
detector in order to get an RGB output to plug into the corresponding pixel.
"""

from typing import List, Tuple, Optional
import numpy as np

def get_color(theta: float) -> np.ndarray:
    """
    Gets RGB value from inputted scalar

    Args:
        theta (float): theta value

    Returns:
        np.ndarray: Corresponding RGB value.
    """
    # assign and reframe theta and colors (horizontal = cool, vertical = hot)
    theta = abs(np.pi / 2 - theta)
    end = np.pi / 2
    BLUE = np.array([0, 0, 255])
    GREEN = np.array([0, 255, 0])
    RED = np.array([255, 0, 0])
    
    # assign color linearly based on angle
    if 0 <= theta <= end/4:
        color = (BLUE * (end/4 - theta) + (BLUE + GREEN) * (theta - 0)) / (end/4)
    elif theta <= 2*end/4:
        color = ((BLUE + GREEN) * (2 * end/4 - theta) + (GREEN) * (theta - end/4)) / (end/4)
    elif theta <= 3*end/4:
        color = (GREEN * (3 * end/4 - theta) + (GREEN + RED) * (theta - 2 * end/4)) / (end/4)
    elif theta <= end:
        color = ((GREEN + RED) * (end - theta) + (RED) * (theta - 3 * end/4)) / (end/4)
    
    return color