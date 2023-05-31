from PIL import Image
import numpy as np

def make_red():
    with Image.open("images/luffy.jpg") as im:
        x=1
        im_arr = np.array(im)
        for row in im_arr:
            for pix in row:
                pix[0] = (x * 255 + pix[0]) // (x + 1)
        new_im = Image.fromarray(im_arr)
        new_im.show()

    
def get_kernel(im_arr, center, sigma):
    x_cen, y_cen = center
    h, w, _ = im_arr.shape
    kernel = np.zeros(shape=(h, w))
    
    for y, row in enumerate(im_arr):
        for x, _ in enumerate(row):
            coef = 1 / (2 * np.pi * sigma**2)
            exp_term = np.exp(- ((x-x_cen)**2 + (y-y_cen)**2) / (2 * sigma**2))
            weight = coef * exp_term
            kernel[y, x] = weight

with Image.open("images/wings_of_freedom.jpg") as im:
    im_arr = np.array(im)
    print(im_arr.shape)
    get_kernel(im_arr, (67, 68), 1)
    