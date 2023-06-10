"""
This is a copy of the greyscale.py file found in the other_algorithms folder
that has been modified so that it takes the strength to be 1, effectively
fully grescaling the image for further use in Canny edge detection.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import click


def greyscale(img_arr: np.ndarray) -> np.ndarray:
    """
    Operates on array representation of image to return a median blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image

    Returns:
        np.ndarray: Array representation of the greyscaled image
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    
    for y, row in enumerate(img_arr):
        for x, pixel in enumerate(row):
            # setting new pixel value to greyscaled RGB from pixel
            new_img_arr[y, x] = np.mean(pixel)
    
    return new_img_arr