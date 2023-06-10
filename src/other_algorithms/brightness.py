"""
Brightness in an image can be changed simply by adding or subtracting a constant
from each RGB value in the image. This is that implementation.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import click


def brighten(img_arr: np.ndarray, brightness_factor: float) -> np.ndarray:
    """
    Operates on array representation of image to return a median blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        brightness_factor (float): factor to change brightness by

    Returns:
        np.ndarray: Array representation of the brightened image
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    
    for y, row in enumerate(img_arr):
        for x, pixel in enumerate(row):
            
            # increasing brightness
            new_pixel = pixel + 255 * brightness_factor
            for i, rgb_val in enumerate(new_pixel):
                if rgb_val < 0:
                    new_pixel[i] = 0
                if rgb_val > 255:
                    new_pixel[i] = 255
            
            # assigning new RGB value to pixel
            new_img_arr[y, x] = new_pixel
    
    return new_img_arr
            

# click commands
@click.command(name="brightness")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-b', '--brightness-factor', type=float, default=0.5)

def change_brightness(filename: str, brightness_factor: float) -> None:
    """
    command
    """
    if not -1 <= brightness_factor <= 1:
        raise ValueError("brightness factor must be between -1 and 1")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = brighten(img_arr, brightness_factor)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    change_brightness()