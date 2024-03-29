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
    ndims = 1

    def __init__(self, key, name):
        super().__init__(key, name, 'utm33n')

    def create_job(self, coords, email, dedicated):
        coords = [util.from_latlon(xy, self.coords) for xy in coords]
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

        try:
            code, response = util.make_request('startExport', params)
            if 'Error'in response:
                return 'Received error: ' + response['Error']
            elif not response.get('Success', False):
                return 'Received error: ' + str(response.get('ErrorMessage', 'Unknown error'))
        except Exception as e:
            return 'Error making request: ' + str(e)

        return response['JobID']


class TiledImageModel(Project):

    supports_email = False
    supports_dedicated = False
    zoomlevels = (2, 20)
    datatype = 'geoimage'
    ndims = 3

    def __init__(self, key, name, url):
        super().__init__(key, name, 'spherical-mercator')
        self.url = url

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
            url = self.url.format(zoom=zoom, x=x, y=y)
            for _ in range(10):
                r = requests.get(url)
                if r.status_code == 200:
                    break
            else:
                print('failed', x, y, zoom, url)

            filename = filesystem.project_file(self.key, f'{zoom}-{x}-{y}.jpeg')
            with open(filename, 'wb') as f:
                f.write(r.content)
            manager.increment_progress()
            filenames.append(filename)

        return filenames
