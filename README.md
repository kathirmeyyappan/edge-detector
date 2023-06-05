# Simple Image Processing Algorithms
This is a collection of some simple image processing algorithms such as Gaussian Blur, imlpemented from scratch in python. Image handling is done with the PIL library and computation is with NumPy.

## Gaussian Blur
Gaussian blur is a blur algorithm which maintains detail well due to assigning weights based on distance from the original pixel. It makes use of the Gaussian function (also known as 'normal distribution' and 'bell curve') to assign weights when blurring per-pixel. When looking at how to convolve a pixel's surrounds to its own new value, we look to the Gaussian function, centered around this pixel in 2 dimensions, to assign weights for how each of the surrounding pixels will contribute to the center pixel's new RGB values. 

This works because when stretching out to infinity, the integral under a Guassian function will be 1.00. Of course, after around 3 standard deviations, the effect is negligible. So, the algorithm that I've attempted to implement from scratch uses a kernel that is constructed from the Gaussian function in 2 dimensions (shown below).

<p align="center">
  <img src="https://i.imgur.com/u0pCJ6q.png" alt="Gaussian Function">
</p>

The kernel is then normalized so the sum of its elements is 1.00 per color and convolved (using the Hadamard product) with the matrix containing the corresponding pixel and its surroundings. This value is stored in the pixel's corresponding spot in the new image. The values are then summed up per color in the RGB 1 x 3 matrix, which finally yields the value for one pixel. This processes is repeated on every pixel of the image. To read more about Gaussian blur, see here: https://en.wikipedia.org/wiki/Gaussian_blur. 

My implementation of the Gaussian blur algorithm is in [gaussian_blur.py](src/gaussian_blur.py). To run this file, run this from the root: ```python3 src/simple_crop_gui.py -f [FILEPATH] -s [SIGMA_VALUE]```, where the filepath is from the root (e.g. ```images/luffy.py```) and sigma is the strength (typically an integer between 1 and 10 inclusive for reasonable results). 

Larger files and sigma values take significantly longer beause of the computationally expensive nature of running Gaussian blur from scratch. Surely, PIL and other image handling libraries utilize advanced optimization techniques. As these implementations are for my own educational purposes and are meant to be semantically understandable, I will leave things as are. Below is an example of running (from the root) ```python3 src/gaussian_blur.py -f images/luffy.jpg -s SIGMA``` for ```SIGMA``` ∈ [1, 2, 3, 4] along with the original image (i.e. ```SIGMA``` = 0).

<p align="center">
  <img src="https://i.imgur.com/YUtJuHR.png" alt="Gaussian Blur Demonstration"></img>
</p>

## Box Blur
Box blur is the most simple blur algorithm. It takes the average RGB values of all pixels within a given distance of the target pixel. Because it uses a simple average, it is quite easy to implement iteratively, where we just take the arithmetic mean of all the pixels in the needed range. This implementation is in [box_blur.py](src/box_blur.py). To run this file, run this from the root: ```python3 src/box_blur.py -f [FILEPATH] -r [RADIUS]```, where the filepath is from the root (e.g. ```images/luffy.py```) and radius corresponds to the strength.

Alternatively, we can use a moving window identically to the Gaussian Blur algorithm where our kernel matrix consists of uniform values to represent identical weights in a "weighted" average. A moving window kernel where our given radius is 1 is shown below:

<p align="center">
  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/a1b3fadc7b147cf0904a66d9521b55df701eafd9" alt="3 x 3 1's Kernel"></img>
</p>

This implementation (made because most of the groundwork was laid in [gaussian_blur.py](src/gaussian_blur.py)) can be found in [box_blur_moving_window.py](src/box_blur_moving_window.py). This file may be run in the same way as [box_blur.py](src/box_blur.py).

Below is an example of running (from the root) ```python3 src/box_blur.py -f images/luffy.jpg -r RADIUS``` for ```RADIUS``` ∈ [3, 6, 9, 12] along with the original image (i.e. ```RADIUS``` = 0). These radius values roughly correspond to the Gaussian blur ```SIGMA``` values of 1, 2, 3, and 4. 

As can be seen above, even though the blur factors are similar, many details are lost in this blur method. So though it is less computationally expensive, it retains much less detail.

## Median Blur
Median blur is very simple to box blur in that it does a very simple operation over a certain range surrounding the pixel of interest to calculate its new value. The difference is that it will take a median for each of the R, G, and B values over the given range rather than their mean. As such, my implementation, which is in [median_blur.py](src/median_blur.py), is very similar to [box_blur.py](src/box_blur.py) save for the calculation of the pixel RGB value (which is a median instead of a mean).

Because we are taking a median, severe outliers do not at all contribute to the image.

## Simple Crop
Using matrices in R^n, it is quite easy to cut out a 'rectangle' using indices. For image cropping, we take the array representation of the image and simply index it as ```img_arr[row_start : row_end + 1, col_start : col_end + 1]```. The implementation for this is in the simple_crop function from [simple_crop.py](src/simple_crop.py).

In [simple_crop_gui.py](src/simple_crop_gui.py), I have made a simple graphical user interface for cropping in pygame. To run the file, run this from the root: ```python3 src/simple_crop_gui.py -f [FILEPATH]```, where the filepath is from the root (e.g. ```images/luffy.py```). For more info on it, try ```python3 src/simple_crop_gui.py --help```. Below is an example use of the GUI.

<p align="center">
  <img src="https://imgur.com/p6SUhVk.gif" alt="Simple Crop Gui Demonstration">
</p>
