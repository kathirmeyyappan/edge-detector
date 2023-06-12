"""
This is a simple program that takes a gif and returns its Sobel counterpart. I'm
just making this to have a pretty header for my README, lol.
"""

from typing import List, Tuple, Optional
import numpy as np
import time
from PIL import Image
import click
from sobel_edge_detector import sobel_edge_detect


def sobel_gif(img_arr_seq: List[np.ndarray]) -> List[np.ndarray]:
    """
    Performs Sobel edge detection on an image array.

    Args:
        img_arr_seq (List[np.ndarray]): sequence of 3-d array representations of 
            images which constitute the gif.

    Returns:
        List[np.ndarray]: Sequence of array representations of the edge 
            detector applied images which constitute the gif
    """
    sobel_seq = []
    for i, img_arr in enumerate(img_arr_seq):
        print(f"\nPROCESSING IMAGE {i+1}/{len(img_arr_seq)}")
        sobel_seq.append(sobel_edge_detect(img_arr))
        print(f"\nAPPLIED CANNY EDGE DETECTION TO IMAGE {i+1}/{len(img_arr_seq)}\n")
    
    return sobel_seq

# click commands
@click.command(name="sobel_gif_maker")
@click.option('-f', '--filename', type=click.Path(exists=True))

def sobel_animate(filename: str) -> None:
    """
    command
    """
    gif_name = filename.split('/')[-1][:-4]
    print(gif_name)
    
    # changing gif into list of image arrays
    img_arr_seq = []
    with Image.open(filename) as gif:
        for n in range(gif.n_frames):
            gif.seek(n)
            img_arr = np.array(gif.convert("RGB"))
            img_arr_seq.append(img_arr)

    # applying Sobel edge detection to all images in list
    sobel_gif_arr = sobel_gif(img_arr_seq)
    # changing each image array in list to image and storing in new list
    img_seq = []
    for img_arr in sobel_gif_arr:
        img_seq.append(Image.fromarray(img_arr))
    
    # saving new gif (from list of images) to canny_animation folder
    img_seq[0].save(f"canny_animations/sobel_{gif_name}.gif", 
                    save_all=True, append_images=img_seq[1:], loop=0)

if __name__ == "__main__":
    sobel_animate()