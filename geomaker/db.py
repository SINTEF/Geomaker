from enum import IntEnum
from operator import methodcaller
import json
import hashlib

from area import area as geojson_area
import requests
import toml
from utm import from_latlon
from xdg import XDG_DATA_HOME, XDG_CONFIG_HOME


# Create database if it does not exist
DATA_ROOT = XDG_DATA_HOME / 'geomaker' / 'database'
POLYGON_ROOT = DATA_ROOT / 'polygons'

for d in [DATA_ROOT, POLYGON_ROOT]:
    if not d.exists():
        d.mkdir()

# Config file
CONFIG_FILE = XDG_CONFIG_HOME / 'geomaker.toml'

# Projects
PROJECTS = [
    ('DTM50', 'Terrain model (50 m)'),
    ('DTM10', 'Terrain model (10 m)'),
    ('DTM1',  'Terrain model (1 m)'),
    ('DOM50', 'Object model (50 m)'),
    ('DOM10', 'Object model (10 m)'),
    ('DOM1',  'Object model (1 m)'),
]

for project, _ in PROJECTS:
    proj_dir = DATA_ROOT / project
    if not proj_dir.exists():
        proj_dir.mkdir()


class Status(IntEnum):
    Nothing = 0
    ExportErrored = 1
    ExportWaiting = 2
    ExportProcessing = 3
    DownloadReady = 4
    Downloaded = 5

    def desc(self):
        return {
            Status.Nothing: 'No export request made',
            Status.ExportErrored: 'Export request errored',
            Status.ExportWaiting: 'Waiting for export to start',
            Status.ExportProcessing: 'Waiting for export to finish',
            Status.DownloadReady: 'Data file ready to download',
            Status.Downloaded: 'Data file available locally',
        }[self]

    def action(self):
        return {
            Status.Nothing: 'Export',
            Status.ExportErrored: 'Export',
            Status.ExportWaiting: 'Refresh',
            Status.ExportProcessing: 'Refresh',
            Status.DownloadReady: 'Download',
            Status.Downloaded: 'View',
        }[self]


def make_request(endpoint, params):
    params = json.dumps(params)
    url = f'https://hoydedata.no/laserservices/rest/{endpoint}.ashx?request={params}'
    print(url)
    response = requests.get(url)
    print(response)
    if response.status_code != 200:
        return response.status_code, None
    return response.status_code, json.loads(response.text)


def unicase_name(poly):
    if isinstance(poly, Polygon):
        poly = poly.name
    return poly.lower()


# Same as built-in bisect_right but with a key argument
def bisect_right(a, x, lo=0, hi=None, key=(lambda x: x)):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if key(x) < key(a[mid]):
            hi = mid
        else:
            lo = mid+1
    return lo


class Config(dict):

    def __init__(self):
        super().__init__()
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                self.update(toml.load(f))

    def verify(self, querier):
        if not 'email' in self:
            querier.message(
                'E-mail address',
                'An e-mail address must be configured to make requests to the Norwegian Mapping Authority',
            )
            self['email'] = querier.query_str('E-mail address', 'E-mail:')
        self.write()

    def write(self):
        with open(CONFIG_FILE, 'w') as f:
            toml.dump(self, f)


class Polygon:

    def __init__(self, data, dbid=None, lfid=None, db=None, write=True):
        self.dbid = dbid
        self.data = data
        self.db = db

        self.lfid = None
        self.set_lfid(lfid)

        if write:
            self.write()

    @classmethod
    def from_file(cls, path, **kwargs):
        with open(path, 'r') as f:
            data = json.load(f)
        for proj in data.setdefault('files', {}).values():
            proj['status'] = Status(proj['status'])
        return cls(data, dbid=path.stem, write=False, **kwargs)

    @classmethod
    def from_dbid(cls, dbid, **kwargs):
        return cls.from_file(dbid + '.json', write=False, **kwargs)

    @property
    def filename(self):
        return POLYGON_ROOT.joinpath(self.dbid + '.json')

    @property
    def geojson(self):
        return self.data['data']

    @property
    def points(self):
        return self.geojson['geometry']['coordinates'][0]

    @property
    def name(self):
        return self.data.setdefault('name', self.dbid)

    @name.setter
    def name(self, value):
        self.data['name'] = value
        self.write()

    @property
    def west(self):
        return min(lon for lon, _ in self.points)

    @property
    def east(self):
        return max(lon for lon, _ in self.points)

    @property
    def south(self):
        return min(lat for _, lat in self.points)

    @property
    def north(self):
        return max(lat for _, lat in self.points)

    @property
    def area(self):
        return geojson_area(self.geojson['geometry'])

    @property
    def files(self):
        return self.data.setdefault('files', {})

    def set_lfid(self, lfid):
        if self.lfid is not None:
            self.db.unlink_lfid(self)
        self.lfid = lfid
        if self.lfid is not None:
            self.db.link_lfid(self)

    def write(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def update(self, data):
        self.data['data'] = data
        self.write()

    def delete(self):
        self.filename.unlink()

    def _project(self, project):
        return self.files.setdefault(project, {'status': Status.Nothing})

    def status(self, project):
        return self._project(project)['status']

    def error(self, project):
        return self._project(project).get('error', None) or 'Unknown error occured'

    def datafile(self, project):
        return DATA_ROOT / project / (self.dbid + '.zip')

    def export(self, project, email):
        # Convert points to UTM 33N integers
        coords = [from_latlon(lat, lon, force_zone_number=33, force_zone_letter='N') for lon, lat in self.points]
        coords = [(int(x), int(y)) for x, y, *_ in coords]
        coords = ';'.join(f'{x},{y}' for x, y in coords)

        params = {
            'CopyEmail': email,
            'Projects': project,
            'CoordInput': coords,
            'InputWkid': 25833, # ETRS89 / UTM zone 33N
            'Format': 5,        # GeoTIFF
            'NHM': 1,           # National altitude models
            'ProjectMerge': 1,  # Don't split up data
        }

        code, response = make_request('startExport', params)
        proj = self._project(project)
        if response is None:
            proj['status'] = Status.ExportErrored
            proj['error'] = f'HTTP code {code}'
        elif 'Error' in response:
            proj['status'] = Status.ExportErrored
            proj['error'] = response['Error']
        elif not response.get('Success', False):
            proj['status'] = Status.ExportErrored
            proj['error'] = None
        else:
            proj['status'] = Status.ExportWaiting
            proj['jobid'] = response.get('JobID', -1)

        self.write()

    def refresh(self, project):
        proj = self._project(project)
        code, response = make_request('exportStatus', {'JobID': proj['jobid']})

        if response is None:
            proj['status'] = Status.ExportErrored
            proj['error'] = f'HTTP code {code}'
        elif response['Status'] == 'new':
            proj['status'] = Status.ExportWaiting
        elif response['Status'] == 'processing':
            proj['status'] = Status.ExportProcessing
        elif response['Status'] == 'complete' and response['Finished']:
            proj['status'] = Status.DownloadReady
            proj['url'] = response['Url']

        self.write()

    def download(self, project):
        proj = self._project(project)
        if not 'url' in proj:
            proj['status'] = Status.ExportErrored
            proj['error'] = None

        response = requests.get(proj['url'])
        if response.status_code != 200:
            proj['status'] = Status.ExportErrored
            proj['error'] = f'HTTP code {response.status_code}'

        with open(self.datafile(project), 'wb') as f:
            f.write(response.content)

        proj['status'] = Status.Downloaded


class Database:

    def __init__(self):
        self.data = []
        self.by_lfid = {}
        self.listeners = []

        self._initialize_db()

    def _initialize_db(self):
        self.data = sorted([
            Polygon.from_file(path, db=self)
            for path in POLYGON_ROOT.joinpath('').glob('*.json')
        ], key=unicase_name)

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        yield from self.data

    def __delitem__(self, dbid):
        poly = self.data[dbid]
        poly.delete()

        del self.data[dbid]
        if poly.lfid is not None:
            del self.by_lfid[poly.lfid]

    def __len__(self):
        return len(self.data)

    def notify(self, obj):
        self.listeners.append(obj)

    def message(self, method, *args):
        caller = methodcaller(method, *args)
        for listener in self.listeners:
            caller(listener)

    def update_name(self, index, name):
        new_index = bisect_right(self.data, name, key=unicase_name)
        poly = self[index]
        poly.name = name

        self.message('before_reset', poly)
        self.data = sorted(self.data, key=unicase_name)
        self.message('after_reset')

    def unlink_lfid(self, poly):
        del self.by_lfid[poly.lfid]

    def link_lfid(self, poly):
        self.by_lfid[poly.lfid] = poly

    def index_of(self, poly=None, lfid=None):
        if poly is None:
            poly = self.by_lfid[lfid]
        return self.data.index(poly)

    def create(self, lfid, name, data):
        dbid = hashlib.sha256(data.encode('utf-8')).hexdigest()
        data = json.loads(data)

        poly = Polygon({'data': data}, dbid=dbid, lfid=lfid, db=self)
        poly.name = name

        index = bisect_right(self.data, poly, key=unicase_name)
        self.message('before_insert', index)
        self.data.insert(index, poly)
        self.message('after_insert')

    def update(self, lfid, data):
        self.by_lfid[lfid].update(json.loads(data))

    def delete(self, lfid):
        index = self.index_of(lfid=lfid)

        self.message('before_delete', index)
        del self[index]
        self.message('after_delete')
