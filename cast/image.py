from skimage import img_as_uint
from skimage.color import rgb2grey
from skimage.io import imread, imsave
from skimage.filters import frangi, median
from skimage.morphology import binary_erosion, binary_closing, disk

from .path import ensure_dir


def read(path):
    return imread(str(path))


def write(image, path):
    ensure_dir(path)
    if image.dtype == bool:
        image = img_as_uint(image)
    return imsave(str(path), image)


def frangi_filter(image):
    if image.ndim == 3:
        image = rgb2grey(image)
    return frangi(image)


def erode(image, times=1):
    eroded = image
    for _ in range(times):
        eroded = binary_erosion(eroded)
    return eroded


def closing(image, strel=disk(2)):
    return binary_closing(image, strel)


def median_filter(image, strel=disk(1)):
    return median(image, strel)
