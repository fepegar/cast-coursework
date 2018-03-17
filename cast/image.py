from skimage.filters import frangi
from skimage.color import rgb2grey
from skimage.io import imread, imsave

from .path import ensure_dir


def read(path):
    return imread(str(path))


def write(image, path):
    ensure_dir(path)
    return imsave(str(path), image)


def frangi_filter(image):
    if image.ndim == 3:
        image = rgb2grey(image)
    return frangi(image)
