"""
This implementation takes all of the pixels in a given "box" when downscaling 
an image so that all pixels still play a roll in the final image. This is 
better than nearest neighbor interpolation because we do not lose details.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click


def box_sampling(img_arr: np.ndarray, resize_factor: float) -> np.ndarray:
    """
    Downscales an array representation of an image by taking the average RGB 
    value from box samples in the original image to retain some details.

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        resize_factor (float): Factor by which image is to be resized

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    og_h, og_w, _ = img_arr.shape
    
    # resizing new image array based on scale
    new_h, new_w = int(og_h * resize_factor), int(og_w * resize_factor)
    new_img_arr = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # creating x lattice points
    x_lattice_points = [0]
    for x in range(new_w):
        lattice_pt =  int(x / new_w * og_w)
        if x_lattice_points[-1] != lattice_pt:
            x_lattice_points.append(lattice_pt)

    # creating y lattice points
    y_lattice_points = [0]
    for y in range(new_w):
        lattice_pt =  int(y / new_h * og_h)
        if y_lattice_points[-1] != lattice_pt:
            y_lattice_points.append(lattice_pt)
    
    # assigning box sample average RGB to corresponding pixel in new image
    for y in range(new_h):
        y_low = y_lattice_points[y]
        y_high = None if y == new_h - 1 else y_lattice_points[y + 1]
        
        for x in range(new_w):
            x_low = x_lattice_points[x]
            x_high = None if x == new_w - 1 else x_lattice_points[x + 1]
            
            # getting box sample RGB values and averaging
            rgb_values = np.reshape(img_arr[y_low:y_high, x_low:x_high], 
                                    (-1, 3))
            new_img_arr[y, x] = np.average(rgb_values, axis=0)
    
    return new_img_arr


# click commands
@click.command(name="box_sampling")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-s', '--resize_factor', type=float, default=0.5)

def resize(filename: str, resize_factor: float) -> None:
    """
    command
    """
    if resize_factor <= 0 or resize_factor >= 1:
        raise ValueError("size factor must be between 0 and 1")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = box_sampling(img_arr, resize_factor)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    resize()