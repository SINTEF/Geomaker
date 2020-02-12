from . import util


class Project:

    def __init__(self, key, name, coords):
        self.key = key
        self.name = name
        self.coords = coords

    def __lt__(self, other):
        return self.key < other.key

    def __str__(self):
        return self.key


class DigitalHeightModel(Project):

    supports_email = True
    supports_dedicated = True
    zoomlevels = None

    def __init__(self, key, name):
        super().__init__(key, name, 'utm33n')

    def create_job(self, coords, email, dedicated):
        coords = [util.convert_latlon(xy, 'utm33n') for xy in coords]
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

    def __init__(self, key, name):
        super().__init__(key, name, 'spherical-mercator')
