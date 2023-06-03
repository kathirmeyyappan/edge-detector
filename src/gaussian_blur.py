"""
This is a from-scratch implementation of the Gaussian Blur function, which
utilizes the array-form of a given image and the standard Gaussian distribution 
in 2 dimensions to evenly blur an image with a kernel matrix. This is a 
computationally expensive operation which will take quite some time to run on 
larger images or with large sigma values.

Note that this file uses click for command line support. You can run it from 
the root with: 'python3 src/filename.py --help' to get started.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image
import click


def gaussian_blur(img_arr: np.ndarray, sigma: int, msg: bool) -> np.ndarray:
    """
    Operates on array representation of image to return a guassian blurred array

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        sigma (int): Standard deviation in guassian distribution. Serves as the
            strength of the blur for our purposes.
        msg (bool): Option to display a progress message every time a row is 
            completed

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    new_img_arr = img_arr.copy()
    kernel = get_kernel(sigma)
    for y, row in enumerate(img_arr):
        if msg:
            print(f"{y}/{img_arr.shape[0]} pixel rows calculated")
        for x, _ in enumerate(row):
            # matching image and kernel piece dimensions
            height, width, _ = img_arr.shape
            img_range = find_range((height, width), (x, y), sigma)
            x_min, x_max = img_range[0]
            y_min, y_max = img_range[1]
            img_piece = img_arr[y_min:y_max+1, x_min:x_max+1]
            kernel_piece = get_kernel_piece(kernel, img_range)
            
            # putting blurred pixel into new image array
            new_img_arr[y, x] = pixel_calculate(kernel_piece, img_piece)
    if msg: 
        print("done!")
    return new_img_arr


def find_range(dimensions: Tuple[int, int], center: Tuple[int, int],
                      sigma: int) -> List[Tuple[int, int]]:
    """
    Finds coordinate range for finding the kernel with a given image size and
        pixel coordinate of interest. Accounts for 3 standard deviations.

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
    return [(max(0, x - 3 * sigma), min(x_max - 1, x + 3 * sigma)),
            (max(0, y - 3 * sigma), min(y_max - 1, y + 3 * sigma))]


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


def get_kernel_piece(kernel: np.ndarray,
                     img_range: List[Tuple[int, int]]) -> np.ndarray:
    """
    Normalizes and crops kernel based on respective pixel position so that its
        dimensions match with the image piece for hadamard product.

    Args:
        kernel (np.ndarray): Generic uncropped kernel.
        img_range (List[Tuple[int, int]]): Coordinate ranges to apply crop.

    Returns:
        np.ndarray: Cropped kernel with weighted values summing to 3.00 (which
            guarantees that each RGB will be weighted properly to 1.00). Will be
            used for convolution.
    """
    x_min, x_max = img_range[0]
    y_min, y_max = img_range[1]
    kernel_piece = kernel.copy()

    # vertical cropping
    if y_max - y_min + 1 < kernel.shape[0]:
        if y_min == 0:
            kernel_piece = kernel_piece[kernel.shape[0] - (y_max - y_min + 1):]
        else:
            kernel_piece = kernel_piece[:y_max - y_min + 1]

    # horizontal cropping
    if x_max - x_min + 1 < kernel.shape[1]:
        if x_min == 0:
            kernel_piece = kernel_piece[:, kernel.shape[1] - (x_max - x_min + 1):]
        else:
            kernel_piece = kernel_piece[:, :x_max - x_min + 1]

    # normalizing kernel values so that total sum is 3.00 (1.00 for each RGB)
    return kernel_piece / np.sum(kernel_piece) * 3


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
        if max(img.arr.shape) > 500:
            raise ValueError("file too large for gaussian blur to be efficient")
        new_img_arr = gaussian_blur(img_arr, sigma_value, progress)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    blur()
