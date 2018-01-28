from skimage.io import imread
from image import convolution
import numpy as np
from matplotlib import pyplot as plt

def to_image(array):
    a_min = np.min(array)
    a_max = np.min(array)
    return ((array - a_min)/float(a_max-a_min))*255

image = imread('./civetta.jpg', as_grey=True)
box_blur_kernel = np.ones((20,20))
result = convolution(image,box_blur_kernel)
plt.imshow(result)
plt.show()
