# Simple Image Processing Algorithms
This is a collection of some simple image processing algorithms such as Gaussian Blur, imlpemented from scratch in python. Image handling is done with the PIL library and computation is with NumPy.

## Gaussian Blur
Among the many blur functions is the Gaussian blur, which makes use of the Gaussian function (also known as 'normal distribution' and 'bell curve') to assign weights when blurring per-pixel. When looking at how to convolve a pixel's surrounds to its own new value, we look to the Gaussian function, centered around this pixel in 2 dimensions, to assign weights for how each of the surrounding pixels will contribute to the center pixel's new RGB values. 

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

## Simple Crop
Using matrices in R^n, it is quite easy to cut out a 'rectangle' using indices. For image cropping, we take the array representation of the image and simply index it as ```img_arr[row_start : row_end + 1, col_start : col_end + 1]```. The implementation for this is in the simple_crop function from [simple_crop.py](src/simple_crop.py).

In [simple_crop.py](src/simple_crop.py), I have made a simple graphical user interface for cropping in pygame. To run the file, run this from the root: ```python3 src/simple_crop_gui.py -f [FILEPATH]```, where the filepath is from the root (e.g. ```images/luffy.py```). For more info on it, try ```python3 src/simple_crop_gui.py --help```. Below is an example use of the GUI.

<p align="center">
  <img src="https://imgur.com/p6SUhVk.gif" alt="Simple Crop Gui Demonstration">
</p>
