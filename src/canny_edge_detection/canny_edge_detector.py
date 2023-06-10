"""
This is my implementation of the Canny edge detection algorithm
"""

from typing import List, Tuple, Optional
import numpy as np
import time
from PIL import Image
import click
from helper_blur import gaussian_blur, median_blur
from helper_greyscale import greyscale


def canny_edge_detection(img_arr: np.ndarray) -> np.ndarray:
    """
    Performs Canny edge detection on an image array.

    Args:
        img_arr (np.ndarray): 3-d array representation of image

    Returns:
        np.ndarray: Array representation of the edge detector applied image
    """
    
    # GREYSCALING IMAGE
    greyscaled_img_arr = greyscale(img_arr)
    print("image greyscaled")
    time.sleep(1)
    
    # PERFORMING NOISE REDUCTION WITH MEDIAN GAUSSIAN BLUR
    # median blur, radius = 1
    median_blurred_img_arr = median_blur(greyscaled_img_arr, radius=1)
    print("image median blurred")
    time.sleep(1)
    # Gaussian blur, sigma = 1
    noise_reduced_img_arr = gaussian_blur(median_blurred_img_arr, sigma=1)
    print("image Gaussian blurred")
    time.sleep(1)
    
    # NEXT STEP
    
    return noise_reduced_img_arr


# click commands
@click.command(name="canny_edge_detector")
@click.option('-f', '--filename', type=click.Path(exists=True))

def edge_detect(filename: str) -> None:
    """
    command
    """

    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = canny_edge_detection(img_arr)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    edge_detect()