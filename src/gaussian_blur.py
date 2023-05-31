from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
import click

def make_red():
    with Image.open("images/luffy.jpg") as im:
        x=1
        im_arr = np.array(im)
        for row in im_arr:
            for pix in row:
                pix[0] = (x * 255 + pix[0]) // (x + 1)
        new_im = Image.fromarray(im_arr)
        new_im.show()


def gaussian_blur(im_arr: np.ndarray, sigma: int) -> np.ndarray:
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
    kernel = get_kernel(sigma)
    for y, row in enumerate(im_arr):
        print(y)
        for x, _ in enumerate(row):
            # matching image and kernel piece dimensions
            h, w, _ = im_arr.shape
            im_range = find_range((h, w), (x, y), sigma)
            x_min, x_max = im_range[0]
            y_min, y_max = im_range[1]
            im_piece = im_arr[y_min:y_max+1, x_min:x_max+1]
            kernel_piece = get_kernel_piece(kernel, im_range)
            # putting blurred pixel into new image array
            new_im_arr[y, x] = pixel_calculate(kernel_piece, im_piece)
    
    return new_im_arr

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
        2 dimensional guassian distribution.

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
                     im_range: List[Tuple[int, int]]) -> np.ndarray:
    """
    Normalizes and crops kernel based on respective pixel position so that its
        dimensions match with the image piece for hadamard product.

    Args:
        kernel (np.ndarray): Generic uncropped kernel.
        im_range (List[Tuple[int, int]]): Coordinate ranges to apply crop.

    Returns:
        np.ndarray: Cropped kernel with weighted values summing to 3.00 (which 
            guarantees that each RGB will be weighted properly to 1.00). Will be 
            used for convolution. 
    """
    x_min, x_max = im_range[0]
    y_min, y_max = im_range[1]
    kernel_piece = kernel.copy()
    
    if y_max - y_min + 1 < kernel.shape[0]:
        if y_min == 0:
            kernel_piece = kernel_piece[:y_max + 1]
        else:
            kernel_piece = kernel_piece[kernel.shape[0] - (y_max - y_min + 1):]
            
    if x_max - x_min + 1 < kernel.shape[1]:
        if x_min == 0:
            kernel_piece = kernel_piece[:, :x_max + 1]
        else:
            kernel_piece = kernel_piece[:, kernel.shape[1] - (x_max - x_min + 1):]

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
@click.option('-s', '--sigma-value', type=int, default=1)

def blur(filename: str, sigma_value: int) -> None:
    with Image.open(filename) as im:
        im.show()
        im_arr: np.ndarray
        im_arr = np.array(im)
        new_im_arr = gaussian_blur(im_arr, sigma_value)
        new_im = Image.fromarray(new_im_arr)
        new_im.show()

if __name__ == "__main__":
    blur()