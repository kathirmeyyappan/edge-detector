from typing import List, Tuple
import numpy as np
from PIL import Image

def simple_crop(img_arr: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> np.ndarray:
    """
    Crops the array representation of an image (rectagularly) given the 
        top-right and bottom-left indices.

    Args:
        img_arr (np.ndarray): Array representation of image.
        top_left (Tuple[int, int]): Top-left index tuple (x, y)
        bottom_right (Tuple[int, int]): Bottom-right index tuple (x, y)

    Returns:
        np.ndarray: Array representation of cropped image.
    """    
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError("First point must be above and to the right of second")
    for x in (x1, x2):
        if not 0 <= x < img_arr.shape[1]:
            raise IndexError("Coordinate is out of bounds")
    for y in (y1, y2):
        if not 0 <= y < img_arr.shape[0]:
            raise IndexError("Coordinate is out of bounds")
    
    return img_arr[y1:y2+1, x1:x2+1]


def crop(filename: str, bounds: List[Tuple[int, int]]) -> None:
    """
    command line operation
    """    
    with Image.open(filename) as img:
        img.show()
        img_arr = np.array(img)
        top_left, bottom_right = bounds
        new_img_arr = simple_crop(img_arr, top_left, bottom_right)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()