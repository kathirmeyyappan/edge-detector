"""
This is my implementation of the Canny edge detection algorithm. Note that the
first 3 steps bear striking similarity to the Sobel edge detection method with
the exception of the calculation of theta, which we use here later on.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click
from helper_blur import gaussian_blur, median_blur
from helper_greyscale import greyscale
from helper_rainbow_fill import get_color


def canny_edge_detect(img_arr: np.ndarray, color: bool) -> np.ndarray:
    """
    Performs Canny edge detection on an image array.

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        color (bool): option to color edges based on edge gradient direction

    Returns:
        np.ndarray: Array representation of the edge detector applied image
    """
    # image array dimensions
    HEIGHT, WIDTH, _ = img_arr.shape
    
    
    ### GREYSCALING IMAGE ###
    
    greyscaled_img_arr = greyscale(img_arr)
    print("\nGREYSCALE APPLIED\n")
    
    
    ### PERFORMING NOISE REDUCTION WITH MEDIAN AND GAUSSIAN BLUR ###
    
    # median blur, radius depends on image size to clear 
    r = 0 if WIDTH < 500 else 1
    median_blurred_img_arr = median_blur(greyscaled_img_arr, radius=r)
    print("\nMEDIAN BLUR APPLIED\n")
    
    # Gaussian blur, sigma = 2
    noise_reduced_img_arr = gaussian_blur(median_blurred_img_arr, sigma=2)
    print("\nGAUSSIAN BLUR APPLIED")
    
    
    ### GRADIENT CALCULATION ###
    
    # turning image array into 2-d array (shows only intensity)
    intensity_arr = noise_reduced_img_arr[:, :, 0]
    
    # X and Y Sobel filters (discrete derivative approximations in dx and dy)
    X_KERNEL = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    Y_KERNEL = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    
    # creating array to be convolved with estimated derivative gradients
    convolved_arr = np.zeros((HEIGHT, WIDTH))
    # creating array to store theta (direction) values for intensity change
    theta_vals = np.zeros((HEIGHT, WIDTH))
    
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
            
            # assignment of intensity change value (from gradient)
            convolved_arr[y, x] = np.sqrt(g_x**2 + g_y**2)
            # specialized arctan function to store direction of gradient change
            theta_vals[y, x] = np.arctan2(g_y, g_x)

    # removing edges that weren't calculated
    convolved_arr = np.squeeze(convolved_arr)
    theta_vals = np.squeeze(theta_vals)
    
    # redirecting theta values to lie only between 0 and 180 (mainting direction)
    theta_vals = theta_vals.astype(np.float64)
    theta_vals[theta_vals < 0] += np.pi
          
    print("\nGRADIENT AND IMAGE ARRAY CONVOLVED\n\nTHETA VALUES CALCULATED")
    
    
    ### APPLYING NON-MAXIMUM SUPRESSION ###
    
    # creating new array to store values from non-maximum suppression
    suppressed_arr = convolved_arr.copy()
    
    # iterating through convolved_arr to find perpendicular points and determine
    # if point is valid edge according to theta and adjacent intensities
    for y, row in enumerate(convolved_arr):
        # ignore edge
        if y == 0 or y == HEIGHT - 1:
            continue
            
        for x, _ in enumerate(row):
            # ignore edge
            if x == 0 or x == WIDTH - 1:
                continue
                
            # finding which adjacent angles to check using adjusted theta
            theta = theta_vals[y, x]
            
            if 0 <= theta < np.pi / 8 or \
            7 * np.pi / 8 <= theta < np.pi:
                adj_1 = convolved_arr[y, x - 1]
                adj_2 = convolved_arr[y, x + 1]
            
            elif np.pi / 8 <= theta <= 3 * np.pi / 8:
                adj_1 = convolved_arr[y - 1, x + 1]
                adj_2 = convolved_arr[y + 1, x - 1]
            
            elif 3 * np.pi / 8 <= theta <= 5 * np.pi / 8:
                adj_1 = convolved_arr[y - 1, x]
                adj_2 = convolved_arr[y + 1, x]
            
            elif 5 * np.pi / 8 <= theta <= 7 * np.pi / 8:
                adj_1 = convolved_arr[y + 1, x + 1]
                adj_2 = convolved_arr[y - 1, x - 1]
                
            # assigning new values to suppressed array if center intensity is
            # larger than neighbors (otherwise 0)
            if convolved_arr[y, x] >= adj_1 and convolved_arr[y, x] >= adj_2:
                suppressed_arr[y, x] = convolved_arr[y, x]
            else:
                suppressed_arr[y, x] = 0
            
    print("\nNON-MAXIMUM SUPPRESSION APPLIED")
    
    
    ### APPLYING DOUBLE THRESHOLD AND HYSTERESIS ###
    
    double_threshold_arr = np.zeros_like(suppressed_arr)
    
    high_th = suppressed_arr.max() * 0.1
    low_th = high_th * 0.5
    strong = 255
    weak = 50
    
    for y, row in enumerate(suppressed_arr):
        for x, intensity in enumerate(row):
            
            if intensity > high_th:
                # strong edge
                double_threshold_arr[y, x] = strong
            elif intensity < low_th:
                # false edge
                double_threshold_arr[y, x] = 0
            else:
                # weak edge
                double_threshold_arr[y, x] = weak
                
    print("\nDOUBLE THRESHOLD APPLIED")
    
    # applying hysteresis
    
    # creating final image array
    final_arr = np.zeros_like(double_threshold_arr)
    # directions to check for strong edge
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y, row in enumerate(double_threshold_arr):
        # ignore edge
        if y == 0 or y == HEIGHT - 1:
            continue
            
        for x, val in enumerate(row):
            # ignore edge
            if x == 0 or x == WIDTH - 1:
                continue
            
            if val == strong or val == 0:
                final_arr[y, x] = val
            
            # determining whether to include weak edge based on neighbors being
            # strong edges    
            else:
                include = False
                for dx, dy in directions:
                    if double_threshold_arr[y+dy, x+dx] == strong:
                        include = True
                        break
                final_arr[y, x] = strong if include else 0
    print("\nHYSTERESIS APPLIED")
                    
    
    ### RETURNING FINAL IMAGE ARRAY ###
    
    final_image_arr = np.zeros((HEIGHT - 1, WIDTH - 1, 3), dtype=np.uint8)
    # putting intensity values from final array back into RGB format
    for y, row in enumerate(final_image_arr):    
        for x, _ in enumerate(row):
            if not color:
                val = final_arr[y, x]
                pixel_val = np.array([val, val, val])
            else:
                # optionally assigning edge color based on gradient slope
                if final_arr[y, x] != 0:
                    pixel_val = get_color(theta_vals[y, x])
                else:
                    pixel_val = np.array([0, 0, 0])
            final_image_arr[y, x] = pixel_val
    print("\nTHETA GRADIENT COLORING COMPLETE")
    print("\nIMAGE COMPLETE")
    return final_image_arr


# click commands
@click.command(name="canny_edge_detector")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option("--color/--no-color", default=True)

def edge_detect(filename: str, color: bool) -> None:
    """
    command
    """
    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = canny_edge_detect(img_arr, color)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    edge_detect()