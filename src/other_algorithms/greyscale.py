"""
Greyscale simple takes the average of all the RGB values for a given pixels and
assigns the new pixel with 3 of these values for R, G, and B respectively. When
this is applied to every pixel in an image, we get the greyscale effect.
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
            

# click commands
@click.command(name="greyscale")
@click.option('-f', '--filename', type=click.Path(exists=True))

def grey(filename: str) -> None:
    """
    command
    """
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = greyscale(img_arr)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    grey()