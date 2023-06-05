"""

"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click


def nearest_neighbor_interpolation(img_arr: np.ndarray, resize_factor: float) -> np.ndarray:
    """
    Resizes an array representation of an image by mapping a scaled array's RGB
        values to their corresponding places on the original array.

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        resize_factor (float): Factor by which image is to be resized

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    h, w, _ = img_arr.shape
    
    # resizing new image array based on scale
    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
    new_img_arr = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # finding corresponding point in scalar interpolation from old image
    # to assign in new image
    for y, row in enumerate(new_img_arr):
        y_corr = int(y / new_h * h)
        for x, _ in enumerate(row):
            x_corr = int(x / new_w * w)
            new_img_arr[y, x] = img_arr[y_corr, x_corr]
    
    return new_img_arr
            

# click commands
@click.command(name="nearest_neighbor_interpolation")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-s', '--resize_factor', type=float, default=2.00)

def resize(filename: str, resize_factor: float) -> None:
    """
    command
    """
    if resize_factor <= 0:
        raise ValueError("size factor must be positive")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = nearest_neighbor_interpolation(img_arr, resize_factor)
        print(type(new_img_arr), new_img_arr.shape)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    resize()