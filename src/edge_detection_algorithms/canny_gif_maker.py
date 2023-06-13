"""
This is a simple program that takes a gif and returns its canny counterpart. I'm
just making this to have a pretty header for my README, lol.
"""

from typing import List, Tuple, Optional
import numpy as np
import time
from PIL import Image
import click
from canny_edge_detector import canny_edge_detect


def canny_gif(img_arr_seq: List[np.ndarray], color: bool) -> List[np.ndarray]:
    """
    Performs Canny edge detection on an image array.

    Args:
        img_arr_seq (List[np.ndarray]): sequence of 3-d array representations of 
            images which constitute the gif.
        color (bool): option to color edges based on edge gradient direction

    Returns:
        List[np.ndarray]: Sequence of array representations of the edge 
            detector applied images which constitute the gif
    """
    canny_seq = []
    for i, img_arr in enumerate(img_arr_seq):
        print(f"\nPROCESSING IMAGE {i+1}/{len(img_arr_seq)}")
        canny_seq.append(canny_edge_detect(img_arr, color))
        print(f"\nAPPLIED CANNY EDGE DETECTION TO IMAGE {i+1}/{len(img_arr_seq)}\n")
    
    return canny_seq

# click commands
@click.command(name="canny_gif_maker")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option("--color/--no-color", default=True)

def canny_animate(filename: str, color: bool) -> None:
    """
    command
    """
    gif_name = filename.split('/')[-1][:-4]
    if color:
        gif_name += "_colored"
    
    # changing gif into list of image arrays
    img_arr_seq = []
    with Image.open(filename) as gif:
        for n in range(gif.n_frames):
            gif.seek(n)
            img_arr = np.array(gif.convert("RGB"))
            img_arr_seq.append(img_arr)
    
    # applying Canny edge detection to all images in list
    canny_gif_arr = canny_gif(img_arr_seq, color)
    # changing each image array in list to image and storing in new list
    img_seq = []
    for img_arr in canny_gif_arr:
        img_seq.append(Image.fromarray(img_arr))
    
    # saving new gif (from list of images) to canny_animation folder
    img_seq[0].save(f"canny_animations/canny_{gif_name}.gif", 
                    save_all=True, append_images=img_seq[1:], loop=0)

if __name__ == "__main__":
    canny_animate()