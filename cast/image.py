import numpy as np
from skimage.io import imread, imsave
from skimage import filters, img_as_uint
from skimage.color import rgb2grey, rgb2hsv, rgb2lab
from skimage.morphology import binary_erosion, binary_closing, disk

from .path import ensure_dir


def read(path):
    return imread(str(path))


def write(image, path):
    ensure_dir(path)
    if image.dtype == bool:
        image = img_as_uint(image)
    return imsave(str(path), image)


def grey2rgb(grey, mask=None):
    #pylint: disable=E1101
    import matplotlib.pyplot as plt
    rgb = plt.cm.viridis(grey / grey.max(), bytes=True)[..., :-1]
    if mask is not None:
        rgb[mask == 0] = 0
    return rgb


def frangi_filter(image):
    if image.ndim == 3:
        image = rgb2grey(image)
    return filters.frangi(image)


def erode(image, times=1):
    eroded = image
    for _ in range(times):
        eroded = binary_erosion(eroded)
    return eroded


def closing(image, strel=disk(2)):
    return binary_closing(image, strel)


def median_filter(image, strel=disk(1)):
    return filters.median(image, strel)


def sobel_filter(image):
    if image.ndim == 3:
        image = rgb2grey(image)
    return filters.sobel(image)


def laplacian_filter(image):
    if image.ndim == 3:
        image = rgb2grey(image)
    return filters.laplace(image)


def hsv_filter(image):
    return rgb2hsv(image)


def lab_filter(image):
    return rgb2lab(image)


def get_confusion_map(array1, array2):
    a = array1.astype(bool).ravel()
    b = array2.astype(bool).ravel()
    TP = (a & b).sum()
    FP = ((a ^ b) & a).sum()
    FN = ((a ^ b) & b).sum()
    TN = a.size - TP - FP - FN
    confusion_map = dict(TP=TP, FP=FP, FN=FN, TN=TN)
    return confusion_map


def to_uint8(image):
    image -= image.min()
    image /= image.max()
    image *= 255
    return image.astype(np.uint8)


def compare(bin1, bin2):
    """bin1 is green, bin2 is magenta"""
    bin1 = bin1.astype(bool) * 255
    bin2 = bin2.astype(bool) * 255
    si, sj = bin1.shape
    rgb = np.zeros((si, sj, 3), np.uint8)
    rgb[..., 0] = bin2
    rgb[..., 1] = bin1
    rgb[..., 2] = bin2
    return rgb
