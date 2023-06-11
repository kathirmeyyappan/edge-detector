"""
This is a collection of blur algorithms (found in blur_algorithms folder) which 
are (possibly) relevant to my implementation of the Canny edge detector. I've 
copied them here for the sake ofhaving all of my content relevant to the Canny 
edge detector in one place.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click


def median_blur(img_arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Operates on array representation of image to return a median blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        radius (int): radius of kernel

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    
    for y, row in enumerate(img_arr):
        if y % 50 == 0:
            print(f"{y}/{img_arr.shape[0]} pixel rows calculated for median blur")
        for x, _ in enumerate(row):
            
            # getting image piece dimensions
            height, width, _ = img_arr.shape
            section_bounds = find_range((height, width), (x, y), radius)
            x_min, x_max = section_bounds[0]
            y_min, y_max = section_bounds[1]
            
            # setting new pixel value to median from image piece
            new_img_arr[y, x] = np.array([np.median(
                img_arr[y_min:y_max, x_min:x_max, c]) for c in range(3)])
    
    return new_img_arr


def gaussian_blur(img_arr: np.ndarray, sigma: int) -> np.ndarray:
    """
    Operates on array representation of image to return a guassian blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    # new image array creation
    new_img_arr = img_arr.copy()
    # creates kernel with weights according to gaussian distribution
    kernel = get_kernel(sigma)
    
    for y, row in enumerate(img_arr):
        if y % 25 == 0:
            print(f"{y}/{img_arr.shape[0]} pixel rows calculated for Gaussian blur")
        for x, _ in enumerate(row):
            
            # getting image_piece dimensions
            height, width, _ = img_arr.shape
            img_range = find_range((height, width), (x, y), 3 * sigma)
            x_min, x_max = img_range[0]
            y_min, y_max = img_range[1]
            img_piece = img_arr[y_min:y_max, x_min:x_max]
            
            # cropping kernel to match image_piece
            x_bounds, y_bounds = crop_kernel(kernel, img_range)
            x_min, x_max = x_bounds
            y_min, y_max = y_bounds
            kernel_piece = kernel[y_min:y_max, x_min:x_max]
            
            # normalizing kernel values so that total sum is 3.00 
            # (1.00 for each RGB)
            kernel_piece = kernel_piece / np.sum(kernel_piece) * 3
            
            # putting blurred pixel into new image array
            new_img_arr[y, x] = pixel_calculate(kernel_piece, img_piece)
    
    return new_img_arr


# below are all of the helpers for gaussian_blur


def find_range(dimensions: Tuple[int, int], center: Tuple[int, int],
                      radius: int) -> List[Tuple[int, int]]:
    """
    Finds coordinate range for finding the kernel with a given image size and
        pixel coordinate of interest. 

    Args:
        dimensions (Tuple[int, int]): Image dimensions (height x width)
        center (Tuple[int, int]): Coordinate of pixel of interest
        radius (int): Distance in each cardinal direction that the kernel would
            extend into. Tells us how far cropping should go.

    Returns:
        List[Tuple[int, int]]: [x-range, y-range]
    """
    y_max, x_max = dimensions
    x, y = center
    
    x_min = 0 if x - radius < 0 else x - radius
    x_max = x_max - 1 if x + radius > x_max - 1 else x + radius
    y_min = 0 if y - radius < 0 else y - radius
    y_max = y_max - 1 if y + radius > y_max - 1 else y + radius   
     
    return [(x_min, x_max + 1), (y_min, y_max + 1)]


def get_kernel(sigma: int) -> np.ndarray:
    """
    Pre-calculates the kernel (which will be used for convolution) using the
        2 dimensional guassian distribution. Creates a matrix that extends for
        3 * sigma in each direction (gaussian weights past that are negligible).

    Args:
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.

    Returns:
        np.ndarray: Generic kernel with guassin distribution in 2-d.
    """
    # creating kernel using 2-d guassian distribution
    kernel = np.zeros((6 * sigma + 1, 6 * sigma + 1, 3))
    for y in range(-3 * sigma, 3 * sigma + 1):
        for x in range(-3 * sigma, 3 * sigma + 1):
            coef = 1 / (2 * np.pi * sigma**2)
            exp_term = np.exp(- (x**2 + y**2) / (2 * sigma**2))
            weight = coef * exp_term
            kernel[y + 3 * sigma, x + 3 * sigma] = np.array([weight] * 3)
    return kernel


def crop_kernel(kernel: np.ndarray, img_range: List[Tuple[int, int]]
                ) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Normalizes and crops kernel based on respective pixel position so that its
        dimensions match with the image piece for hadamard product.

    Args:
        kernel (np.ndarray): Generic uncropped kernel.
        img_range (List[Tuple[int, int]]): Coordinate ranges to apply crop.

    Returns:
        np.ndarray: Coordinate ranges within the kernel to apply crop in 
            order to get kernel_piece
    """
    x_min, x_max = img_range[0]
    y_min, y_max = img_range[1]
    new_x_min = new_x_max = new_y_min = new_y_max = None

    # vertical cropping
    if y_max - y_min < kernel.shape[0]:
        if y_min == 0:
            new_y_min, new_y_max = kernel.shape[0] - (y_max - y_min), None
        else:
            new_y_min, new_y_max = None, y_max - y_min

    # horizontal cropping
    if x_max - x_min < kernel.shape[1]:
        if x_min == 0:
            new_x_min, new_x_max = kernel.shape[1] - (x_max - x_min), None
        else:
            new_x_min, new_x_max = None, x_max - x_min

    return [(new_x_min, new_x_max), (new_y_min, new_y_max)]    


def pixel_calculate(kernel: np.ndarray, og_img: np.ndarray) -> np.ndarray:
    """
    Calculates RGB value for given pixel based on identically dimensioned
        kernel and original image. Utilizes Hadamard product before we take the
        pseudo-sum to obtain 1x3 array for RGB.

    Args:
        kernel (np.ndarray): Kernel to apply convolution from.
        og_img (np.ndarray): Original image array

    Returns:
        np.ndarray: RGB value of pixel
    """
    had_product = kernel * og_img
    pixel = np.zeros((1, 3))
    for row in had_product:
        for rgb_piece in row:
            pixel += rgb_piece
    return pixel

### REMOVE LATER

# click commands
@click.command(name="gaussian_blur")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-s', '--sigma-value', type=int, default=2)
@click.option("--progress/--hide-progress", default=True)

def blur(filename: str, sigma_value: int, progress: bool) -> None:
    """
    command line operation
    """
    if type(sigma_value) is not int and not 1 <= sigma_value <= 10:
        raise ValueError("sigma value must be int from 1 to 10")
    
    with Image.open(filename) as img:
        img_arr = np.array(img)
        if max(img_arr.shape) > 500:
            raise ValueError("Image is too large for gaussian blur to be "
                             "efficient. Try another image file or another "
                             "algorithm like box blur")
        new_img_arr = gaussian_blur(img_arr, sigma_value, progress)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()
