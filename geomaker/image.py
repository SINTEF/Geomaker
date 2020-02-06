from PIL import Image
from matplotlib import cm as colormap
import numpy as np


def list_colormaps():
    return colormap.cmap_d.keys()


def array_to_image(data, fmt, cmap, filename):
    maxval = np.max(data)
    if maxval > 0:
        data /= np.max(data)
    data = getattr(colormap, cmap)(data, bytes=True)
    if data.shape[-1] == 4 and fmt == 'jpeg':
        data = data[..., :3]
    image = Image.fromarray(data)
    image.save(str(filename), format=fmt.upper())
