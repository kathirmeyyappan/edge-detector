"""
This image uses bilinear interpolation to resize images while filling in
discrete block gaps with linearly calculated RGB values for pixels in order to
reduce lost detail and blockiness that nearest neighbor interpolation causes.
The general approach is to find "lattice" points from the nearest neighbor
interpolation method, using those as footholds to interpolate between in our
new image. I first interpolate vertically because it is computationally more
efficient because of rows being the outermost layer in our array. Then, I
rotate the image on its side to interpolate along the horizontal axis of
the image. Rotating it back and returning the image, we now have a more
acceptable upscaled image.
"""

from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import click
from nearest_neighbor_interpolation import nearest_neighbor_interpolation


def bilinear_interpolation(img_arr: np.ndarray, resize_factor: float) -> np.ndarray:
    """
    Resizes an array representation of an image by mapping a scaled array's RGB
        values to a linear map where scale is maintained.

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        resize_factor (float): Factor by which image is to be resized

    Returns:
        np.ndarray: Array representation of the blurred image
    """
    og_h, og_w, _ = img_arr.shape
    # getting resized img_arr values
    new_h, new_w = int(og_h * resize_factor), int(og_w * resize_factor)

    # nearest neighbor interpolation image to apply bilinear interpolation on
    new_img_arr = nearest_neighbor_interpolation(img_arr, resize_factor)

    # creating a set of lattice x-coordinates in scaled NNI image
    x_lattice_points = []
    for x in range(new_w):
        if x == 0 or x == new_w - 1 or \
        int(x / new_w * og_w) > int((x - 1) / new_w * og_w):
            x_lattice_points.append(x)

    # creating a set of lattice y-coordinates in scaled NNI image
    y_lattice_points = []
    for y in range(new_h):
        if y == 0 or y == new_h - 1 or \
        int(y / new_h * og_h) > int((y - 1) / new_h * og_h):
            y_lattice_points.append(y)

    # interpolating vertically and rotating onto side
    vertical_linear_interpolation(new_img_arr, y_lattice_points, "vertical")
    new_img_arr = np.rot90(new_img_arr)

    # interpolating vertically (horizontally with respect to old orientation)
    # and rotating back
    vertical_linear_interpolation(new_img_arr, x_lattice_points, "horizontal")
    new_img_arr = np.rot90(np.rot90(np.rot90(new_img_arr)))

    return new_img_arr


def vertical_linear_interpolation(img_arr: np.ndarray, lattice_points: List[int],
                                  orientation: str) -> None:
    """
    Applies bilinear interpolation along the vertical axis in an image array,
        effectively blurring the lines between edges in the image vertically.
        When applied to a rotated image, it will blur horizontally before we
        rotate back.

    Args:
        img_arr (np.ndarray): 3-d array representation of image
        lattice_points (List[int]): List of lattice points along the vertical
            axis that will be used as endpoints for weighted averages applied
            to RGB values.
        orientation (str): absolute name of axis along which we are
            interpolating
    """
    h = img_arr.shape[0]
    index = 0
    for y, row in enumerate(img_arr):
        print(f"{y}/{h} pixel-lines calculated along {orientation} axis")
        # bottom edge of image
        if y == h - 1:
            break
        # move to next lattice point
        if y == h - 1 or y == lattice_points[index + 1]:
            index += 1
        for x, _ in enumerate(row):
            # creating weighted average parameters
            top, bottom = lattice_points[index:index + 2]
            bottom_weight = (y - top) / (bottom - top)
            top_weight = (bottom - y) / (bottom - top)

            #assigning weighted average RGB value to position
            img_arr[y, x] = img_arr[top, x] * top_weight + \
            img_arr[bottom, x] * bottom_weight
    print("done!")


# click commands
@click.command(name="nearest_neighbor_interpolation")
@click.option('-f', '--filename', type=click.Path(exists=True))
@click.option('-s', '--resize_factor', type=float, default=2.00)

def resize(filename: str, resize_factor: float) -> None:
    """
    command
    """
    if resize_factor <= 1:
        raise ValueError("size factor must be greater than 1")

    with Image.open(filename) as img:
        img_arr = np.array(img)
        new_img_arr = bilinear_interpolation(img_arr, resize_factor)
        new_img = Image.fromarray(new_img_arr)
        new_img.show()

if __name__ == "__main__":
    resize()