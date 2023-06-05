"""
Median blur takes the median R, G, and B values in the range specified by the 
blur radius. Here, I have used the find_range function from gaussian_blur.py
to get the range. The code is very similar to box_blur.py except that the 
pixel value is found by taking the median in the range instead of the mean.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import click
from gaussian_blur import find_range


def median_blur(img_arr: np.ndarray, radius: int, msg: bool) -> np.ndarray:
    """
    Operates on array representation of image to return a median blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        radius (int): radius of kernel
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
            
            # getting image piece dimensions
            height, width, _ = img_arr.shape
            section_bounds = find_range((height, width), (x, y), radius)
            x_min, x_max = section_bounds[0]
            y_min, y_max = section_bounds[1]
            
            # setting new pixel value to median from image piece
            new_img_arr[y, x] = np.array([np.median(
                img_arr[y_min:y_max, x_min:x_max, c]) for c in range(3)])
    
    if msg:
        print("done!")
    return new_img_arr
            

# click commands
@click.command(name="median_blur")
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
        new_img_arr = median_blur(img_arr, radius, progress)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()