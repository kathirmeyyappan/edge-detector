"""
This is my implementation of the Sobel edge detection algorithm
"""

from typing import List, Tuple, Optional
import numpy as np
import time
from PIL import Image
import click
from helper_blur import gaussian_blur, median_blur
from helper_greyscale import greyscale


def canny_edge_detection(img_arr: np.ndarray) -> np.ndarray:
    """
    Performs Canny edge detection on an image array.

    Args:
        img_arr (np.ndarray): 3-d array representation of image

    Returns:
        np.ndarray: Array representation of the edge detector applied image
    """
    # 
    HEIGHT, WIDTH, _ = img_arr.shape
    
    
    # GREYSCALING IMAGE
    
    greyscaled_img_arr = greyscale(img_arr)
    print("IMAGE GREYSCALED")
    time.sleep(1)
    
    
    # PERFORMING NOISE REDUCTION WITH MEDIAN AND GAUSSIAN BLUR
    
    # median blur, radius depends on image size to clear 
    r = 0 if WIDTH < 500 else 1
    median_blurred_img_arr = median_blur(greyscaled_img_arr, radius=r)
    print("IMAGE MEDIAN BLURRED")
    time.sleep(1)
    
    # Gaussian blur, sigma = 1
    noise_reduced_img_arr = gaussian_blur(median_blurred_img_arr, sigma=2)
    print("IMAGE GAUSSIAN BLURRED")
    time.sleep(1)
    
    
    # GRADIENT CALCULATION
    
    # turning image array into 2-d array
    intensity_arr = noise_reduced_img_arr[:, :, 0]
    
    # X and Y Sobel filters (discrete derivative approximations in dx and dy)
    X_KERNEL = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Y_KERNEL = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # creating array to be convolved with estimated derivative gradients
    convolved_arr = intensity_arr.copy()
    
    # iterating through intensity_arr to fill convolved array
    for y, row in enumerate(intensity_arr):
        # ignore edge
        if y == 0 or y == HEIGHT - 1:
            continue
        
        for x, _ in enumerate(row):
            # ignore edge
            if x == 0 or x == WIDTH - 1:
                continue
            
            # get Hadamard product and array sum to calculate intensity change
            g_x = np.sum(X_KERNEL * intensity_arr[y-1:y+2, x-1:x+2])
            g_y = np.sum(Y_KERNEL * intensity_arr[y-1:y+2, x-1:x+2])
            
            convolved_arr[y,x] = np.sqrt(g_x**2 + g_y**2)
    print("GRADIENT AND IMAGE ARRAY CONVOLVED")
    time.sleep(1)
    
    # removing edges that weren't convolved
    convolved_arr = np.squeeze(convolved_arr)
    
    # creating final image array 
    final_image = np.zeros((HEIGHT - 1, WIDTH - 1, 3), dtype=np.uint8)
    # putting intensity values from convolved array back into RGB format
    for y, row in enumerate(final_image):        
        for x, g in enumerate(row):
            val = convolved_arr[y,x]
            final_image[y, x] = np.array([val, val, val])
    
    print("DONE")
    return final_image


# click commands
@click.command(name="canny_edge_detector")
@click.option('-f', '--filename', type=click.Path(exists=True))

def edge_detect(filename: str) -> None:
    """
    command
    """
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = canny_edge_detection(img_arr)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    edge_detect()