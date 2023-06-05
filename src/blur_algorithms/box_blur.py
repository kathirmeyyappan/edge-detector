"""
This implementation of box blur simply takes a mean of all of the pixel RGB 
values within a certain range of the target pixel. We use the find_range 
function from gaussian_blur.py to deal with indexing edges of the image.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import click
from gaussian_blur import find_range


def box_blur(img_arr: np.ndarray, radius: int, msg: bool) -> np.ndarray:
    """
    Operates on array representation of image to return a box blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        radius (int): radius of kern
        msg (bool): Option to display a progress message every time a row is 
            completed

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    
    for y, row in enumerate(img_arr):
        if msg: 
            print(f"{y}/{img_arr.shape[0]} pixel rows calculated")
        for x, _ in enumerate(row):
            
            # getting image_piece dimensions
            height, width, _ = img_arr.shape
            section_bounds = find_range((height, width), (x, y), radius)
            x_min, x_max = section_bounds[0]
            y_min, y_max = section_bounds[1]
            
            # setting new pixel value to range average from image
            surround_count = (x_max - x_min) * (y_max - y_min)
            pixel_rgb_sum = np.zeros(3)
            for row in img_arr[y_min:y_max, x_min:x_max]:
                for pix in row:
                    pixel_rgb_sum += pix
            new_img_arr[y, x] = pixel_rgb_sum / surround_count
    
    if msg:
        print("done!")
    return new_img_arr
            

# click commands
@click.command(name="box_blur")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-r', '--radius', type=int, default=2)
@click.option("--progress/--hide-progress", default=True)

def blur(filename: str, radius: int, progress: bool) -> None:
    """
    command
    """
    if type(radius) is not int:
        raise ValueError("radius value must be int")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = box_blur(img_arr, radius, progress)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()