"""
This implementation of box blur (which is just a blur algorithm that uses a
simple mean to calculate pixels) is built upon my implementation of Gaussian
blur. This is because all functions other than the get_kernel function are just
generic matrix operations that convolute regardless of the kind of matrix.
Because the "mean" operation is the same as using a kernel with all identical
weights, I went with this implementation. It should be noted that here, even
the main function is almost identical to the gaussian_blur function, with the
only difference being the value of the pre-set kernel.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import click
from gaussian_blur import find_range, get_kernel_piece, pixel_calculate

def box_blur(img_arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Operates on array representation of image to return a box blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        radius (int): radius of kern

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    # assigns equal weights to all pixels in range to take a normal mean
    kernel = np.ones((2 * radius + 1, 2 * radius + 1, 3))
    
    for y, row in enumerate(img_arr):
        for x, _ in enumerate(row):
            # matching image and kernel piece dimensions
            height, width, _ = img_arr.shape
            img_range = find_range((height, width), (x, y), radius)
            x_min, x_max = img_range[0]
            y_min, y_max = img_range[1]
            img_piece = img_arr[y_min:y_max+1, x_min:x_max+1]
            kernel_piece = get_kernel_piece(kernel, img_range)
            
            # putting blurred pixel into new image array
            new_img_arr[y, x] = pixel_calculate(kernel_piece, img_piece)
    
    return new_img_arr


# click commands
@click.command(name="box_blur")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-r', '--radius', type=int, default=2)

def blur(filename: str, radius: int) -> None:
    """
    command line operation
    """
    if type(radius) is not int:
        raise ValueError("radius value must be int")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = box_blur(img_arr, radius)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()
