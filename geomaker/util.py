from io import BytesIO
from itertools import chain
import json

from humanfriendly import format_size
import numpy as np
from pygeotile.point import Point as PyGeoPoint
from pygeotile.tile import Tile as PyGeoTile
import requests
from utm import from_latlon


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
        return response.status_code, None
    return response.status_code, json.loads(response.text)


def download_streaming(url, mgr):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        return None
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


def convert_latlon(point, coords):
    if coords == 'latlon':
        return point
    elif coords.startswith('utm'):
        zonenum = int(coords[3:-1])
        zoneletter = coords[-1].upper()
        x, y, *_ = from_latlon(point[1], point[0], force_zone_number=33, force_zone_letter='N')
        if isinstance(point[0], np.ndarray):
            return x, y
        return np.array([x, y])
    elif coords == 'spherical-mercator':
        pt = PyGeoPoint.from_latitude_longitude(point[1], point[0])
        return np.array([*pt.meters])
    raise ValueError(f'Unknown coordinate system: {coords}')


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
    points = [convert_latlon(point, 'spherical-mercator') for point in points]
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
