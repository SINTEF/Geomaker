import ctypes as ct
import hashlib
import triangle
from pathlib import Path

from matplotlib import cm as colormap
import numpy as np

from .asynchronous import async_job


def has_support(fmt):
    try:
        if is_image_format(fmt):
            import PIL
        elif fmt == 'g2':
            import splipy
        elif fmt == 'stl':
            import stl
        elif fmt in ('vtk', 'vtu', 'vts'):
            import vtk
    except ImportError:
        return False
    return True

def supports_texture(fmt):
    return fmt in ('vtk', 'vtu', 'vts', 'obj')

def supports_structured_choice(fmt):
    return fmt in ('vtk',)

def is_image_format(fmt):
    # Note: GeoTIFF is not an image format
    return fmt in ('png', 'jpeg')

def requires_rectangle(fmt):
    return is_image_format(fmt) or fmt == 'tiff'

def is_structured_format(fmt):
    """Return true if FMT is inherently structured. If the format can
    support unstructured AND structured grids, return false. In that
    case, supports_structured_choice should return true.
    """
    return is_image_format(fmt) or fmt in ('vts', 'g2', 'tiff')


CATEGORIZED_MAPS = [
    {
        'title': 'Perceptually Uniform Sequential',
        'entries': {
            'Viridis': 'viridis', 'Plasma': 'plasma',
            'Inferno': 'inferno', 'Magma': 'magma',
            'Cividis': 'cividis',
        },
    },
    {
        'title': 'Sequential',
        'entries': {
            'Greys': 'Greys',
            'Purples': 'Purples',
            'Blues': 'Blues',
            'Greens': 'Greens',
            'Oranges': 'Oranges',
            'Reds': 'Reds',
            'Orange-Red': 'OrRd',
            'Purple-Red': 'PuRd',
            'Red-Purple': 'RdPu',
            'Blue-Purple': 'BuPu',
            'Purple-Blue': 'PuBu',
            'Green-Blue': 'GnBu',
            'Blue-Green': 'BuGn',
            'Yellow-Green': 'YlGn',
            'Yellow-Orange-Brown': 'YlOrBr',
            'Yellow-Orange-Red': 'YlOrRd',
            'Yellow-Green-Blue': 'YlGnBu',
            'Purple-Blue-Green': 'PuBuGn',
        },
    },
    {
        'title': 'Sequential (2)',
        'entries': {
            'Gray': 'gray',
            'Bone': 'bone',
            'Pink': 'pink',
            'Spring': 'spring',
            'Summer': 'summer',
            'Autumn': 'autumn',
            'Winter': 'winter',
            'Hot': 'hot',
            'Cool': 'cool',
            'Wistia': 'Wistia',
            'Copper': 'copper',
        },
    },
    {
        'title': 'Diverging',
        'entries': {
            'Pink-Green': 'PiYG',
            'Purple-Green': 'PRGn',
            'Brown-Blue-Green': 'BrBG',
            'Orange-Purple': 'PuOr',
            'Red-Gray': 'RdGy',
            'Red-Blue': 'RdBu',
            'Red-Yellow-Blue': 'RdYlBu',
            'Red-Yellow-Green': 'RdYlGn',
            'Blue-White-Red': 'bwr',
            'Spectral': 'Spectral',
            'Cool-Warm': 'coolwarm',
            'Seismic': 'seismic',
        },
    },
    {
        'title': 'Cyclic',
        'entries': {
            'Twilight': 'twilight',
            'Twilight (shifted)': 'twilight_shifted',
            'Hue': 'hsv',
        },
    },
    {
        'title': 'Miscellaneous',
        'entries': {
            'Cube Helix': 'cubehelix',
            'Gnuplot': 'gnuplot',
            'Gnuplot 2': 'gnuplot2',
            'Jet': 'jet',
            'Ocean': 'ocean',
            'Rainbow': 'rainbow',
            'Terrain': 'terrain',
            'GIST Earth': 'gist_earth',
            'GIST Stern': 'gist_stern',
            'GIST Rainbow': 'gist_rainbow',
        },
    },
]

COLORMAPS = {
    name: value
    for category in CATEGORIZED_MAPS
    for name, value in category['entries'].items()
}

def iter_map_categories():
    for category in CATEGORIZED_MAPS:
        filtered_keys = [key for key, name in category['entries'].items() if name in colormap.cmap_d]
        yield category['title'], filtered_keys

def get_colormap(name, invert=False):
    name = COLORMAPS[name]
    if invert:
        name += '_r'
    return getattr(colormap, name)

def preview_colormap(name, res=100, invert=False):
    data = np.linspace(0, 1, res)[np.newaxis, :]
    data = get_colormap(name, invert=invert)(data, bytes=True)
    return data


def array_to_image(data, fmt, filename, cmap=None, invert=False):
    if data.shape[-1] == 1:
        maxval = np.max(data)
        if maxval > 0:
            data /= np.max(data)
        data = get_colormap(cmap, invert=invert)(data[...,0], bytes=True)
    else:
        data = data.astype(np.uint8)
    if data.shape[-1] == 4 and fmt == 'jpeg':
        data = data[..., :3]
    data = data[:,::-1,:].transpose((1,0,2))
    from PIL import Image
    image = Image.fromarray(data)
    image.save(str(filename), format=fmt.upper())


@async_job()
def export_job(*args, **kwargs):
    return export(*args, **kwargs)


def export(polygon, project, manager, boundary_mode='exterior',
           rotation_mode='none', coords='utm33n', resolution=None,
           maxpts=None, format='png', structured=False,
           colormap='Terrain', invert=False, texture=False,
           zero_sea_level=True, filename=None, directory=None,
           axis_align=False, offset_origin=False):

    # Sanitize parameters
    image_mode = is_image_format(format)
    if not supports_texture(format):
        texture = False
    if not supports_structured_choice(format):
        structured = is_structured_format(format)

    if format == 'tiff' and coords != 'utm33n':
        raise ValueError('GeoTIFF output for other formats than UTM33N is not supported')

    manager.report_max(4 if texture else 3)

    manager.report_message('Generating geometry')
    if structured:
        (in_x, in_y), (out_x, out_y), trf = polygon.generate_meshgrid(
            boundary_mode, rotation_mode, in_coords=project.coords,
            out_coords=coords, resolution=resolution, maxpts=maxpts,
            axis_align=axis_align,
        )
    else:
        (in_x, in_y), (out_x, out_y), tri = polygon.generate_triangulation(
            in_coords=project.coords, out_coords=coords, resolution=resolution,
        )
    manager.increment_progress()

    if texture:
        manager.report_message('Generating texture coordinates')
        left = np.min(out_x)
        right = np.max(out_x)
        down = np.min(out_y)
        up = np.max(out_y)
        uvcoords = np.stack([(out_x - left) / (right - left), (out_y - down) / (up - down)], axis=out_x.ndim)
        manager.increment_progress()

    if offset_origin:
        out_x -= offset_origin[0]
        out_y -= offset_origin[1]

    manager.report_message('Generating data')
    data = polygon.interpolate(project, in_x, in_y)
    if not zero_sea_level:
        data -= np.min(data)
    manager.increment_progress()

    manager.report_message('Saving file')
    if filename is None:
        filename = hashlib.sha256(data.data).hexdigest() + '.' + format
        filename = Path(directory) / filename

    if image_mode:
        array_to_image(data, format, filename, cmap=colormap, invert=invert)
    elif format == 'g2':
        from splipy import Surface, BSplineBasis
        from splipy.io import G2
        cpts = np.stack([out_x, out_y, data[...,0]], axis=2)
        knots = [[0.0] + list(map(float, range(n))) + [float(n-1)] for n in data.shape[:2]]
        bases = [BSplineBasis(order=2, knots=kts) for kts in knots]
        srf = Surface(*bases, cpts, raw=True)
        with G2(filename) as g2:
            g2.write(srf)
    elif format == 'stl':
        from stl.mesh import Mesh as STLMesh
        mesh = STLMesh(np.zeros(tri.shape[0], STLMesh.dtype))
        mesh.vectors[:,:,0] = out_x[tri]
        mesh.vectors[:,:,1] = out_y[tri]
        mesh.vectors[:,:,2] = data[tri,0]
        mesh.save(filename)
    elif format in ('vtk', 'vtu', 'vts', 'obj'):
        import vtk
        import vtk.util.numpy_support as vtknp

        pointsarray = np.stack([out_x.flat, out_y.flat, data.flat], axis=1)
        points = vtk.vtkPoints()
        points.SetData(vtknp.numpy_to_vtk(pointsarray))

        if structured:
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions(*out_x.shape[::-1], 1)
        else:
            ncells = len(tri)
            cellarray = np.concatenate([3*np.ones((ncells, 1), dtype=int), tri], axis=1, dtype=vtknp.ID_TYPE_CODE)
            cells = vtk.vtkCellArray()
            cells.SetCells(ncells, vtknp.numpy_to_vtkIdTypeArray(cellarray.ravel()))
            grid = vtk.vtkUnstructuredGrid()
            grid.SetCells(vtk.VTK_TRIANGLE, cells)

        grid.SetPoints(points)

        if texture:
            grid.GetPointData().SetTCoords(vtknp.numpy_to_vtk(uvcoords.reshape(-1, 2)))

        if format == 'vts':
            writer = vtk.vtkXMLStructuredGridWriter()
        elif format == 'vtu':
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif format == 'obj':
            writer = vtk.vtkOBJWriter()
            geofilter = vtk.vtkGeometryFilter()
            geofilter.SetInputData(grid)
            geofilter.Update()
            grid = geofilter.GetOutput()
        else:
            writer = vtk.vtkStructuredGridWriter() if structured else vtk.vtkUnstructuredGridWriter()

        writer.SetFileName(filename)
        writer.SetInputData(grid)
        writer.Write()
    elif format == 'tiff':
        import tifffile
        tifffile.imwrite(
            filename, data[:,::-1].T,
            extratags=[
                # GeoKeyDirectoryTag
                (34735, 'h', 28,
                 (1, 1, 1, 6,             # Version and number of geo keys
                  1024, 0, 1, 1,          # GTModelTypeGeoKey (2D projected CRS)
                  1025, 0, 1, 1,          # GTRasterTypeGeoKey (pixels denote areas)
                  2048, 0, 1, 4258,       # GeodeticCRSGeoKey (ETRS89)
                  2050, 0, 1, 6258,       # GeodeticDatumGeoKey (ETRS89)
                  2051, 0, 1, 8901,       # PrimeMeridianGeoKey (Greenwich)
                  3072, 0, 1, 25833,      # ProjectedCRSGeoKey (ETRS89: UTM33N)
                  ), True),
                # ModelTransformationTag
                (34264, 'd', 16, tuple(trf.flat), True),
            ]
        )

    manager.increment_progress()

    return filename


class TriangulateIO(ct.Structure):
    _fields_ = [
        ('pointlist', ct.POINTER(ct.c_double)),
        ('pointattributelist', ct.POINTER(ct.c_double)),
        ('pointmarkerlist', ct.POINTER(ct.c_int)),
        ('numberofpoints', ct.c_int),
        ('numberofpointattributes', ct.c_int),
        ('trianglelist', ct.POINTER(ct.c_int)),
        ('triangleattributelist', ct.POINTER(ct.c_double)),
        ('trianglearealist', ct.POINTER(ct.c_double)),
        ('neighborlist', ct.POINTER(ct.c_int)),
        ('numberoftriangles', ct.c_int),
        ('numberofcorners', ct.c_int),
        ('numberoftriangleattributes', ct.c_int),
        ('segmentlist', ct.POINTER(ct.c_int)),
        ('segmentmarkerlist', ct.POINTER(ct.c_int)),
        ('numberofsegments', ct.c_int),
        ('holelist', ct.POINTER(ct.c_double)),
        ('numberofholes', ct.c_int),
        ('regionlist', ct.POINTER(ct.c_double)),
        ('numberofregions', ct.c_int),
        ('edgelist', ct.POINTER(ct.c_int)),
        ('edgemarkerlist', ct.POINTER(ct.c_int)),
        ('normlist', ct.POINTER(ct.c_double)),
        ('numberofedges', ct.c_int),
    ]


def triangulate(points, segments, max_area=None, verbose=False):

    options = 'pzjq'
    if verbose:
        options += 'VV'
    else:
        options += 'Q'
    if max_area:
        options += f'a{max_area}'

    mesh = triangle.triangulate({'vertices':points, 'segments':segments}, options)

    return np.array(mesh['vertices'].tolist()), np.array(mesh['triangles'].tolist())
