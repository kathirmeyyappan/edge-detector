from PIL import Image
import numpy as np
from typing import List, Tuple, Optional

def make_red():
    with Image.open("images/luffy.jpg") as im:
        x=1
        im_arr = np.array(im)
        for row in im_arr:
            for pix in row:
                pix[0] = (x * 255 + pix[0]) // (x + 1)
        new_im = Image.fromarray(im_arr)
        new_im.show()


def find_kernel_range(dimensions: Tuple[int, int], center: Tuple[int, int], 
                      sigma: int) -> List[Tuple[int, int]]:
    """
    Finds coordinate range for finding the kernel with a given image size and 
        pixel coordinate of interest. Accounts for 4 standard deviations.

    Args:
        dimensions (Tuple[int, int]): Image dimensions (height x width)
        center (Tuple[int, int]): Coordinate of pixel of interest
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.

    Returns:
        List[Tuple[int, int]]: [x-range, y-range]
    """
    y_max, x_max = dimensions
    x, y = center
    return [(max(0, x - 4 * sigma), min(x_max, x + 4 * sigma)), 
            (max(0, y - 4 * sigma), min(y_max, y + 4 * sigma))]
    
    
def get_kernel(im_arr: np.ndarray, 
               center: Tuple[int, int], sigma: int) -> np.ndarray:
    """
    Calculates the kernel (which will be used for convolution) using the
        2 dimensional guassian distribution.

    Args:
        im_arr (np.ndarray): 3-d array representation of image.
        center (Tuple[int, int]): Coordinate for pixel that we are calculating 
            the kernel for.
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.
    
    Returns:
        np.ndarray: Kernel with weighted values summing to 3.00 (which 
            guarantees that each RGB will be weighted properly to 1.00). Will be 
            used for convolution. 
    """
    # creating kernel using 2-d guassian distribution
    x_cen, y_cen = center
    kernel_range = find_kernel_range((im_arr.shape[0], im_arr.shape[1]), 
                                     center, sigma)
    print(center, kernel_range)
    x_min, x_max = kernel_range[0]
    y_min, y_max = kernel_range[1]
    kernel = np.zeros(((y_max - y_min + 1), (x_max - x_min + 1), 3))
    
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            coef = 1 / (2 * np.pi * sigma**2)
            exp_term = np.exp(- ((x-x_cen)**2 + (y-y_cen)**2) / (2 * sigma**2))
            weight = coef * exp_term
            kernel[y, x] = np.array([weight] * 3)
    # normalizing kernel values so that total sum is 3.00 (1.00 for each RGB)
    return kernel / np.sum(kernel) * 3


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


def guassian_blur(im_arr: np.ndarray, sigma: int) -> np.ndarray:
    """
    Operates on array representation of image to return a guassian blurred array

    Args:
        im_arr (np.ndarray): 3-d array representation of image
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    new_im_arr = im_arr.copy()
    for y, row in enumerate(im_arr):
        for x, _ in enumerate(row):
            kernel = get_kernel(im_arr, (x, y), sigma)
            # matching image portion location and dimensions to kernel
            h, w, _ = im_arr.shape
            im_range = find_kernel_range((h, w), (x, y), sigma)
            x_min, x_max = im_range[0]
            y_min, y_max = im_range[1]
            im_section = im_arr[y_min:y_max+1, x_min:x_max+1]
            # putting blurred pixel into new image array
            new_im_arr[y, x] = pixel_calculate(kernel, im_section)
    return new_im_arr


with Image.open("images/wings_of_freedom.jpg") as im:
    im.show()
    im_arr = np.array(im)
    blurred_im_arr = guassian_blur(im_arr, 3)
    blurred_image = Image.fromarray(blurred_im_arr)
    blurred_image.show()
    