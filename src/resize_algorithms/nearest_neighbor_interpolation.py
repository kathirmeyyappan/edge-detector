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
    raise NotImplementedError

# click commands
@click.command(name="nearest_neighbor_interpolation")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-s', '--resize_factor', type=float, default=2.00)

def blur(filename: str, resize_factor: float) -> None:
    """
    command
    """
    if float <= 0:
        raise ValueError("size factor must be positive")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = nearest_neighbor_interpolation(img_arr, resize_factor)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()