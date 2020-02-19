from io import BytesIO
from itertools import chain
import json
import sys

from humanfriendly import format_size
import numpy as np
from pygeotile.point import Point as PyGeoPoint
from pygeotile.tile import Tile as PyGeoTile
import requests
import utm
import tifffile as tif
import tifffile.tifffile_geodb as geodb


class SingletonMeta(type):
    """Metaclass for creating singleton classes."""

    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)
        return cls.__instances[cls]


def make_request(endpoint, params):
    """Submit a request to hoydedata.no at the given endpoint.
    'Params' should be a dict of parameters.  Returns a tuple with
    HTTP status code and potentially a dict of results.
    """
    params = json.dumps(params)
    url = f'https://hoydedata.no/laserservices/rest/{endpoint}.ashx?request={params}'
    response = requests.get(url)
    if response.status_code != 200:
        print(response.text, file=sys.stderr)
        raise Exception(f'HTTP status code {response.status_code}. See stderr for more information.')
    return response.status_code, json.loads(response.text)


def download_streaming(url, mgr):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(response.text, file=sys.stderr)
        raise Exception(f'HTTP status code {response.status_code}. See stderr for more information.')
    nbytes = int(response.headers['Content-Length'])
    mgr.report_max(nbytes)
    responsedata = BytesIO()
    down = 0
    for chunk in response.iter_content(16384):
        responsedata.write(chunk)
        down += len(chunk)
        mgr.increment_progress(len(chunk))
        mgr.report_message('Downloading · {}/{}'.format(
            format_size(down, keep_width=True),
            format_size(nbytes, keep_width=True)
        ))
    return responsedata


EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = 2.0 * np.pi * EARTH_RADIUS / 2.0


def from_latlon(point, coords):
    if coords == 'latlon':
        x, y = point
    elif coords.startswith('utm'):
        zonenum = int(coords[3:-1])
        zoneletter = coords[-1].upper()
        x, y, *_ = utm.from_latlon(
            point[1], point[0], force_zone_number=zonenum, force_zone_letter=zoneletter
        )
    elif coords == 'spherical-mercator':
        lon, lat = point
        x = lon * ORIGIN_SHIFT / 180
        y = np.log(np.tan((90 + lat) * np.pi / 360)) / np.pi * ORIGIN_SHIFT
    else:
        raise ValueError(f'Unknown coordinate system: {coords}')
    if isinstance(x, np.ndarray):
        return x, y
    return np.array([x, y])


def to_latlon(point, coords):
    if coords == 'latlon':
        lon, lat = point
    elif coords.startswith('utm'):
        zonenum = int(coords[3:-1])
        zoneletter = coords[-1].upper()
        lat, lon = utm.to_latlon(
            point[0], point[1], zone_number=zonenum, zone_letter=zoneletter
        )
    elif coords == 'spherical-mercator':
        x, y = point
        lon = x / ORIGIN_SHIFT * 180
        lat = y / ORIGIN_SHIFT * 180
        lat = 180 / np.pi * (2 * np.arctan(np.exp(lat * np.pi / 180)) - np.pi / 2)
    else:
        raise ValueError(f'Unknown coordinate system: {coords}')
    if isinstance(lon, np.ndarray):
        return lon, lat
    return np.array([lon, lat])


def _tile_at(point, zoom):
    """Return the map tile located at the given point in Spherical
    Mercator coordinates with the specified zoom level.
    """
    point = PyGeoPoint.from_meters(point[0], point[1])
    tile = PyGeoTile.for_point(point, zoom)
    return tile.google


def _are_neighbors(lft, rgt):
    """Check if two tiles are neighbors in the plus-or-x sense."""
    return abs(lft[0] - rgt[0]) + abs(lft[1] - rgt[1]) <= 1


def _find_border_tiles(a, b, lft, rgt, zoom, tileset, tol=1e-5):
    """Given a point and tile a, lft and a point and tile b, rgt, add
    all the intermediate tiles to the tileset, up to a given tolerance.
    """
    midpt = (a + b) / 2
    mid = _tile_at(midpt, zoom)
    tileset.add(mid)
    if lft != mid and not _are_neighbors(lft, mid) and np.linalg.norm(a - midpt) > tol:
        _find_border_tiles(a, midpt, lft, mid, zoom, tileset, tol=tol)
    if rgt != mid and not _are_neighbors(rgt, mid) and np.linalg.norm(b - midpt) > tol:
        _find_border_tiles(b, midpt, rgt, mid, zoom, tileset, tol=tol)


def _plus_tiles(tile):
    """Yield all the plus-type neighbors of a given tile."""
    x, y = tile
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _diag_tiles(tile, with_bridge=False):
    """Yield all the x-type neighbors of a given tile, together with
    the two plus-type neighbors bridging the gap for each.
    """
    x, y = tile
    if with_bridge:
        yield (x + 1, y + 1), (x + 1, y), (x, y + 1)
        yield (x - 1, y + 1), (x - 1, y), (x, y + 1)
        yield (x + 1, y - 1), (x + 1, y), (x, y - 1)
        yield (x - 1, y - 1), (x - 1, y), (x, y - 1)
    else:
        yield (x + 1, y + 1)
        yield (x - 1, y + 1)
        yield (x + 1, y - 1)
        yield (x - 1, y - 1)


def _disconnect_tiles(tileset):
    """Assuming the given tileset is separable into two connected
    subsets, compute this separation.
    """
    start_tile, *new_tiles = iter(tileset)
    new_tiles = set(new_tiles)
    part_a = {start_tile}
    queue = [start_tile]
    while queue:
        tile = queue.pop(0)
        for neighbor in _plus_tiles(tile):
            if neighbor in part_a:
                continue
            if neighbor in new_tiles:
                part_a.add(neighbor)
                queue.append(neighbor)

    part_b = tileset - part_a
    return part_a, part_b


def geotiles(points, zoom):
    """Compute a set of tiles covering the given polygon (in latitude
    and longitude coordinates) at the given zoom level.
    """
    points = [from_latlon(point, 'spherical-mercator') for point in points]
    tileset = set()

    # Find all the border tiles
    for a, b in zip(points[:-1], points[1:]):
        lft = _tile_at(a, zoom)
        rgt = _tile_at(b, zoom)
        tileset.add(lft)
        tileset.add(rgt)
        _find_border_tiles(a, b, lft, rgt, zoom, tileset)

    # Extend the border region to eliminate purely diagonal connections
    new_tiles = set()
    for tile in tileset:
        for diag_tile, neighbor_a, neighbor_b in _diag_tiles(tile, with_bridge=True):
            if not diag_tile in tileset:
                continue
            if neighbor_a not in tileset and neighbor_b not in tileset:
                new_tiles.add(neighbor_a)
                new_tiles.add(neighbor_b)
    tileset |= new_tiles

    # Fatten the border in both directions
    new_tiles = set()
    for tile in tileset:
        for neighbor in chain(_plus_tiles(tile), _diag_tiles(tile)):
            if neighbor not in tileset:
                new_tiles.add(neighbor)

    # Separate the new tiles in two connected groups, and pick the
    # smallest, which should be the interior
    part_a, part_b = _disconnect_tiles(new_tiles)
    if len(part_a) > len(part_b):
        border = part_b
    else:
        border = part_a

    # Iteratively fill in the interior
    while border:
        tileset |= border
        new_border = set()
        for tile in border:
            for neighbor in _plus_tiles(tile):
                if neighbor not in tileset:
                    new_border.add(neighbor)
        border = new_border

    return tileset


def verify_geotiff(filename):
    """Runs some basic checks on a (presumed) GeoTIFF file to verify
    that some assumptions we have are correct. The GeoTIFF file format
    is too varied to support in its entirety.
    """

    with tif.TiffFile(filename) as tiff:
        assert tiff.is_geotiff
        metadata = tiff.geotiff_metadata

        i, j, k, x, y, z = metadata['ModelTiepoint']
        assert i == j == k == z == 0

        rx, ry, rz = metadata['ModelPixelScale']
        assert rz == 0
        assert rx == ry

        assert metadata['GTModelTypeGeoKey'] == geodb.ModelType.Projected
        assert metadata['GTRasterTypeGeoKey'] == geodb.RasterPixel.IsArea
        assert metadata['GeogGeodeticDatumGeoKey'] == geodb.Datum.European_Reference_System_1989
        assert metadata['GeogPrimeMeridianGeoKey'] == geodb.PM.Greenwich
        assert metadata['GeogPrimeMeridianLongGeoKey'] == 0
        assert metadata['ProjectedCSTypeGeoKey'] == 25833   # UTM33N
        assert metadata['ProjLinearUnitsGeoKey'] == geodb.Linear.Meter
        assert metadata['VerticalUnitsGeoKey'] == geodb.Linear.Meter
