from PIL import Image
from matplotlib import cm as colormap
import numpy as np


def list_colormaps():
    return colormap.cmap_d.keys()


def array_to_image(data, cmap, true_zero, filename):
    if not true_zero:
        data -= np.min(data)
    maxval = np.max(data)
    if maxval > 0:
        data /= np.max(data)
    image = Image.fromarray(getattr(colormap, cmap)(data, bytes=True))
    image.save(str(filename))
