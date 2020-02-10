import hashlib
from pathlib import Path

from matplotlib import cm as colormap
import numpy as np
from PIL import Image
from splipy import Surface, BSplineBasis
from splipy.io import G2
from stl.mesh import Mesh as STLMesh

from .asynchronous import async_job


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


def is_image_format(fmt):
    return fmt in {'png', 'jpeg'}


@async_job()
def export_job(*args, **kwargs):
    return export(*args, **kwargs)


def export(polygon, project, manager, boundary_mode='exterior',
           rotation_mode='none', coords='utm33n', resolution=None,
           maxpts=None, format='png', colormap='terrain',
           zero_sea_level=True, filename=None, directory=None):

    print('Exporting in', __import__('threading').get_ident())
    manager.report_max(3)

    image_mode = is_image_format(format)
    if not image_mode:
        boundary_mode = 'actual'
        rotation_mode = 'none'

    manager.report_message('Generating geometry')
    if format == 'stl':
        x, y, tri = polygon.generate_triangulation(coords, resolution)
    else:
        x, y = polygon.generate_meshgrid(
            boundary_mode, rotation_mode, coords,
            resolution=resolution, maxpts=maxpts
        )
    manager.increment_progress()

    manager.report_message('Generating data')
    if image_mode:
        data = polygon.interpolate(project, x, y)
    else:
        data = polygon.interpolate(project, y, x)
    if not zero_sea_level:
        data -= np.min(data)
    manager.increment_progress()

    manager.report_message('Saving file')
    if filename is None:
        filename = hashlib.sha256(data.data).hexdigest() + '.' + format
        filename = Path(directory) / filename

    if image_mode:
        array_to_image(data, format, colormap, filename)
    elif format == 'g2':
        cpts = np.stack([x, y, data], axis=2)
        knots = [[0.0] + list(map(float, range(n))) + [float(n-1)] for n in data.shape]
        bases = [BSplineBasis(order=2, knots=kts) for kts in knots]
        srf = Surface(*bases, cpts, raw=True)
        with G2(filename) as g2:
            g2.write(srf)
    elif format == 'stl':
        mesh = STLMesh(np.zeros(tri.shape[0], STLMesh.dtype))
        mesh.vectors[:,:,0] = x[tri]
        mesh.vectors[:,:,1] = y[tri]
        mesh.vectors[:,:,2] = data[tri]
        mesh.save(filename)
    manager.increment_progress()

    return filename
