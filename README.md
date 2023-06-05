# Simple Image Processing Algorithms
This is a collection of some simple image processing algorithms, imlpemented from scratch in python. Image handling is done with the PIL library and computation is with NumPy.

## Table of Contents
 - [Gaussian Blur](https://github.com/kathirmeyyappan/simple-image-processing-algorithms/#gaussian-blur)

 - [Box Blur](https://github.com/kathirmeyyappan/simple-image-processing-algorithms/#box-blur)

 - [Median Blur](https://github.com/kathirmeyyappan/simple-image-processing-algorithms/#median-blur)

 - [Simple Crop](https://github.com/kathirmeyyappan/simple-image-processing-algorithms/#simple-crop)

## Gaussian Blur
Gaussian blur is a blur algorithm which maintains detail well due to assigning weights based on distance from the original pixel. It makes use of the Gaussian function (also known as 'normal distribution' and 'bell curve') to assign weights when blurring per-pixel. When looking at how to convolve a pixel's surrounds to its own new value, we look to the Gaussian function, centered around this pixel in 2 dimensions, to assign weights for how each of the surrounding pixels will contribute to the center pixel's new RGB values. 

This works because when stretching out to infinity, the integral under a Guassian function will be 1.00. Of course, after around 3 standard deviations, the effect is negligible. So, the algorithm that I've attempted to implement from scratch uses a kernel that is constructed from the Gaussian function in 2 dimensions (shown below).

<p align="center">
  <img src="https://i.imgur.com/u0pCJ6q.png" alt="Gaussian Function">
</p>

The kernel is then normalized so the sum of its elements is 1.00 per color and convolved (using the Hadamard product) with the matrix containing the corresponding pixel and its surroundings. This value is stored in the pixel's corresponding spot in the new image. The values are then summed up per color in the RGB 1 x 3 matrix, which finally yields the value for one pixel. This processes is repeated on every pixel of the image. To read more about Gaussian blur, see [here](https://en.wikipedia.org/wiki/Gaussian_blur). 

My implementation of the Gaussian blur algorithm is in [gaussian_blur.py](src/blur_algorithms/gaussian_blur.py). To run this file, run this from the root: ```python3 src/resize_algorithms/simple_crop_gui.py -f [FILEPATH] -s [SIGMA_VALUE]```, where the filepath is from the root (e.g. ```images/luffy.py```) and sigma is the strength (typically an integer between 1 and 10 inclusive for reasonable results). 

Larger files and sigma values take significantly longer beause of the computationally expensive nature of running Gaussian blur from scratch. Surely, PIL and other image handling libraries utilize advanced optimization techniques. As these implementations are for my own educational purposes and are meant to be semantically understandable, I will leave things as are. Below is an example of running (from the root) ```python3 src/blur_algorithms/gaussian_blur.py -f images/luffy.jpg -s SIGMA``` for ```SIGMA``` ∈ [1, 2, 3, 4] along with the original image (i.e. ```SIGMA``` = 0).

<p align="center">
  <img src="https://i.imgur.com/YUtJuHR.png" alt="Gaussian Blur Demonstration"></img>
</p>

## Box Blur
Box blur is the most simple blur algorithm. It takes the average RGB values of all pixels within a given distance of the target pixel. Because it uses a simple average, it is quite easy to implement iteratively, where we just take the arithmetic mean of all the pixels in the needed range. This implementation is in [box_blur.py](src/blur_algorithms/box_blur.py). To run this file, run this from the root: ```python3 src/blur_algorithms/box_blur.py -f [FILEPATH] -r [RADIUS]```, where the filepath is from the root (e.g. ```images/luffy.py```) and and radius corresponds to the range from which we take the mean.

Alternatively, we can use a moving window identically to the Gaussian Blur algorithm where our kernel matrix consists of uniform values to represent identical weights in a "weighted" average. To learn more about box blur, see [here](https://en.wikipedia.org/wiki/Box_blur). A moving window kernel where our given radius is 1 is shown below:

<p align="center">
  <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/a1b3fadc7b147cf0904a66d9521b55df701eafd9" alt="3 x 3 1's Kernel"></img>
</p>

This implementation (made because most of the groundwork was laid in [gaussian_blur.py](src/blur_algorithms/gaussian_blur.py)) can be found in [box_blur_moving_window.py](src/blur_algorithms/box_blur_moving_window.py). This file may be run in the same way as [box_blur.py](src/blur_algorithms/box_blur.py).

Below is an example of running (from the root) ```python3 src/blur_algorithms/blur_algorithms/box_blur.py -f images/luffy.jpg -r RADIUS``` for ```RADIUS``` ∈ [3, 6, 9, 12] along with the original image (i.e. ```RADIUS``` = 0). These radius values roughly correspond to the Gaussian blur ```SIGMA``` values of 1, 2, 3, and 4 because their respective kernel sizes are equal in my implementation (which used a standard deviation of 3 for the Gaussian distribution kernel).

<p align="center">
  <img src="https://i.imgur.com/WquQVC4.png" alt="Box Blur Demonstration"></img>
</p>

As can be seen above, even though the blur factors are similar, many details are lost in this blur method compared to Gaussian blur. So though it is less computationally expensive, it retains much less detail.

## Median Blur
Median blur is very similar to box blur in that it does a very simple operation over a certain range surrounding the pixel of interest to calculate its new value. The difference is that it will take a median for each of the R, G, and B values over the given range rather than their mean. As such, my implementation, which is in [median_blur.py](src/blur_algorithms/median_blur.py), is very similar to [box_blur.py](src/blur_algorithms/box_blur.py) save for the calculation of the pixel RGB value (which is a median instead of a mean). To run this file, run this from the root: ```python3 src/blur_algorithms/median_blur.py -f [FILEPATH] -r [RADIUS]```, where the filepath is from the root (e.g. ```images/luffy.py```) and radius corresponds to the range from which we take the median.

Because we are taking a median, severe outliers do not at all contribute to the image. Sporadic RGB values (which often correspond to "wrong" color pixels in given ranges) are completely ignored in the median calculating process so long as the majority of values in the range are consistent in their RGB values. As such, this blur method is very good at removing unwanted noise from images. Shown below is [forgers_noise.jpg](images/forgers_noise.jpg) (the result of running [forgers.jpg](images/forgers.jpg) through an [image noise adder](https://pinetools.com/add-noise-image)) before and after running median blur with radius 2 (run ```python3 src/blur_algorithms/median_blur.py -f images/forgers_noise.jpg -r 2``` from root).

<p align="center">
  <img src="https://i.imgur.com/7cLBfnt.png" alt="Monochronic Noise Median Blur Demonstration">
</p>

As can be seen above, there is actually a quite remarkable amount of noise correction that seems to occur from simply taking the median RGB values. Running median blur on the same image with noise that is not monochronic still yields quite good results, as seen below (run ```python3 src/blur_algorithms/median_blur.py -f images/forgers_noise_2.jpg -r 2``` from root).

<p align="center">
  <img src="https://i.imgur.com/pKDpGAx.jpg" alt="Multichronic Noise Median Blur Demonstration">
</p>

We can compare this to other blur algorithms to show its supremacy in removing noise. Below (from top to bottom) are the median blur, Gaussian blur, and box blur algorithms with ```RADIUS``` (```SIGMA``` for Gaussian blur) values of 0, 1, 2, and 3. As can be seen, median blur completely eliminates noise almost immediately, though edge clarity is lost. Still, even with ```RADIUS``` = 1, median blur outperforms the other algorithms greatly.

<p align="center">
  <img src="https://i.imgur.com/TNq0KZq.png" alt="Noisy Luffy Blur Demonstration" width="734" height="650">
</p>

It should also be noted that due to our calculation of median not requiring a moving window of any sort, this algorithm is very fast when compared with Gaussian blur, for instance.

## Simple Crop
Using matrices in R^n, it is quite easy to cut out a 'rectangle' using indices. For image cropping, we take the array representation of the image and simply index it as ```img_arr[row_start : row_end + 1, col_start : col_end + 1]```. The implementation for this is in the simple_crop function from [simple_crop.py](src/resize_algorithms/simple_crop.py).

In [simple_crop_gui.py](src/resize_algorithms/simple_crop_gui.py), I have made a simple graphical user interface for cropping in pygame. To run the file, run this from the root: ```python3 src/simple_crop_gui.py -f [FILEPATH]```, where the filepath is from the root (e.g. ```images/luffy.py```). For more info on it, try ```python3 src/resize_algorithms/simple_crop_gui.py --help```. Below is an example use of the GUI.

<p align="center">
  <img src="https://imgur.com/p6SUhVk.gif" alt="Simple Crop Gui Demonstration">
</p>
