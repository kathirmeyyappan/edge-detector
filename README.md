# Simple Image Processing Algorithms
This is a collection of some simple image processing algorithms such as Gaussian Blur, imlpemented from scratch in python. Image handling is done with the PIL library and computation is with NumPy.

## Simple Crop
Using matrices in R^n, it is quite easy to cut out a 'rectangle' using indices. For image cropping, we take the array representation of the image and simply index it as ```img_arr[row_start : row_end + 1, col_start : col_end + 1]```. The implementation for this is in the simple_crop function from [src/simple_crop.py](simple_crop.py).

In [src/simple_crop_gui.py](simple_crop_gui.py), I have made a simple graphical user interface for cropping in pygame. To run the file, run this from the root: ```python3 src/simple_crop_gui.py -f [FILEPATH]```, where the filepath is from the root (e.g. ```images/luffy.py```). For more info on it, try ```python3 src/simple_crop_gui.py --help```. Below is an example use of the GUI.

![Simple Crop Gui Demonstration](https://imgur.com/p6SUhVk.gif)
