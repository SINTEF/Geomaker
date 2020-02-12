import requests

from . import util, filesystem
from .asynchronous import async_job


class Project:

    def __init__(self, key, name, coords):
        self.key = key
        self.name = name
        self.coords = coords

    def __lt__(self, other):
        return self.key < other.key

    def __str__(self):
        return self.key

    def create_job(self, coords, **kwargs):
        """Should return either an int (a job ID) or an AbstractJob object.
        If the latter, that job should, when run, return a list of
        filenames and the project object itself.
        """
        raise NotImplementedError


class DigitalHeightModel(Project):

    supports_email = True
    supports_dedicated = True
    zoomlevels = None
    datatype = 'geotiff'

    def __init__(self, key, name):
        super().__init__(key, name, 'utm33n')

    def create_job(self, coords, email, dedicated):
        coords = [util.convert_latlon(xy, self.coords) for xy in coords]
        coords = ';'.join(f'{int(x)},{int(y)}' for x, y in coords)
        params = {
            'CopyEmail': email,
            'Projects': self.key,
            'CoordInput': coords,
            'ProjectMerge': 1 if dedicated else 0,
            'InputWkid': 25833,      # ETRS89 / UTM zone 33N
            'Format': 5,             # GeoTIFF,
            'NHM': 1,                # National altitude models
        }

        code, response = util.make_request('startExport', params)
        if response is None:
            return f'HTTP code {code}'
        elif 'Error'in response:
            return response['Error']
        elif not response.get('Success', False):
            return 'Unknown error'

        return response['JobID']


class TiledImageModel(Project):

    supports_email = False
    supports_dedicated = False
    zoomlevels = (2, 16)
    datatype = 'geoimage'

    def __init__(self, key, name):
        super().__init__(key, name, 'latlon')

    def create_job(self, coords, zoom, dedicated):
        assert dedicated == False
        return TiledImageModel._download_tiles(self=self, coords=coords, zoom=zoom)

    @staticmethod
    @async_job(message='Downloading tiles')
    def _download_tiles(self, coords, zoom, manager):
        tiles = util.geotiles(coords, zoom)
        manager.report_max(len(tiles))

        filenames = []
        for x, y in tiles:
            url = f'https://kartverket.maplytic.no/tile/_nib/{zoom}/{x}/{y}.jpeg'
            r = requests.get(url)
            if r.status_code != 200:
                continue
            filename = filesystem.project_file(self.key, f'{zoom}-{x}-{y}.jpeg')
            with open(filename, 'wb') as f:
                f.write(r.content)
            manager.increment_progress()

        return filenames
