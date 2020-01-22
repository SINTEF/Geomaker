from operator import methodcaller
import json
import hashlib

from area import area as geojson_area
from xdg import XDG_DATA_HOME


# Create database if it does not exist
DATA_ROOT = XDG_DATA_HOME / 'geomaker' / 'database'
POLYGON_ROOT = DATA_ROOT / 'polygons'

for d in [DATA_ROOT, POLYGON_ROOT]:
    if not d.exists():
        d.mkdir()


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


class Polygon:

    def __init__(self, data, dbid=None, lfid=None, db=None, write=True):
        self.dbid = dbid
        self.data = data
        self.db = db

        self.lfid = None
        self.set_lfid(lfid)

        if write:
            self.write()

    @property
    def filename(self):
        return POLYGON_ROOT.joinpath(self.dbid + '.json')

    @property
    def points(self):
        return self.data['geometry']['coordinates'][0]

    @property
    def name(self):
        return self.data.get('name', self.dbid)

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
        return geojson_area(self.data['geometry'])

    def _copy_special_keys(self, data):
        for key in ['name']:
            data[key] = self.data[key]

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
        self._copy_special_keys(data)
        self.data = data
        self.write()

    def delete(self):
        self.filename.unlink()

    @classmethod
    def from_file(cls, path, **kwargs):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data, dbid=path.stem, write=False, **kwargs)

    @classmethod
    def from_dbid(cls, dbid, **kwargs):
        return cls.from_file(dbid + '.json', write=False, **kwargs)


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
        data['name'] = name

        poly = Polygon(data, dbid=dbid, lfid=lfid, db=self)
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
