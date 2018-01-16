from skimage import data
from skimage.transform import pyramid_gaussian, resize
from skimage.io import imread
from matplotlib import pyplot as plt

def plot(y):
    plt.plot(y)
    plt.show()

def im_read(fname, as_grey=False):
    """
        args
            fname : string
            as_grey : boolean
        returns
            img_array : ndarray
    """
    img_array = imread(fname, as_grey=as_grey)
    return img_array

def resize_to(img_array, output_shape):
    """
        args
            img_array : ndarray
            output_shape : int list
        returns
            img_array : ndarray
    """
    img_array = resize(img_array, output_shape, mode='constant')
    return img_array

def gpyramid(img_array, downscale=2, max_levels=-1):
    """
        args
            img_array : ndarray
            downscale : int
            max_layer : int
        return
            gpyramid : ndarray list
    """
    rows, cols, dim = img_array.shape
    gpyramid = list(pyramid_gaussian(img_array,
                        max_layer=max_levels,
                        downscale=downscale))
    return gpyramid

def show_img(img_array):
    """
        args
            img_array : ndarray
    """
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    plt.show()