from contextlib import contextmanager
from functools import lru_cache, partial, wraps
import hashlib
from itertools import tee, islice
from io import BytesIO
import json
from math import ceil
from operator import methodcaller
from pathlib import Path
from zipfile import ZipFile

from area import area as geojson_area
from bidict import bidict
import numpy as np
import tifffile as tif
import requests
import toml
from utm import from_latlon
from xdg import XDG_DATA_HOME, XDG_CONFIG_HOME

from . import polyfit, image

import sqlalchemy as sql
import sqlalchemy.orm as orm
from sqlalchemy.ext.declarative import declarative_base

DeclarativeBase = declarative_base()


# Create database if it does not exist
DATA_ROOT = XDG_DATA_HOME / 'geomaker'
THUMBNAIL_ROOT = DATA_ROOT / 'thumbnails'

for d in [DATA_ROOT, THUMBNAIL_ROOT]:
    if not d.exists():
        d.mkdir()

# Config file
CONFIG_FILE = XDG_CONFIG_HOME / 'geomaker.toml'

# Data file
DATA_FILE = DATA_ROOT / 'geomaker.toml'

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


def download_geotiffs(url, project, dedicated):
    """Download a zip-file with GeoTIFF data, in standard hoydedata.no format.
    This unzips the file and saves all the TIFFs to disk, but does not
    update the database. Return a list of filenames.
    """
    response = requests.get(url)
    if response.status_code != 200:
        return None

    filenames = []
    with ZipFile(BytesIO(response.content), 'r') as z:
        tifpaths = [path for path in z.namelist() if path.endswith('.tif')]
        if dedicated:
            assert len(tifpaths) == 1

        for path in tifpaths:
            data = z.read(path)
            if dedicated:
                filename = hashlib.sha256(data).hexdigest() + '.tiff'
            else:
                filename = Path(path).stem.split('_', 1)[-1] + '.tiff'
            filename = DATA_ROOT / project / filename
            with open(filename, 'wb') as f:
                f.write(data)
            filenames.append(filename)
    return filenames


def convert_latlon(point, coords):
    if coords == 'latlon':
        return point
    elif coords.startswith('utm'):
        zonenum = int(coords[3:-1])
        zoneletter = coords[-1].upper()
        x, y, *_ = from_latlon(point[1], point[0], force_zone_number=33, force_zone_letter='N')
        return np.array([x, y])
    raise ValueError(f'Unknown coordinate system: {coords}')


class AsyncWorker:
    """Utility class representing a asynchronous package of work.  The
    'work' argument is a function to be called in a separate thread,
    and the 'callback' argument is a function to be called with the
    return value of the 'work' function.

    This object is suitable as an argument the main GUI run_thread
    function, as it returns its own callback partially applied with
    the result of the work package.

    Thus:
        a(b())

    is the same as:
        worker = AsyncWorker(a, b)
        retval = worker()    # calls a()
        retval()             # calls b(...)
    """

    def __init__(self, work, callback):
        self.work = work
        self.callback = callback

    def __call__(self):
        retval = self.work()
        return partial(self.callback, retval)


def asynchronous(func):
    """Wrap a function with an optional 'asynchronous' keyword argument.
    The inner function should return a tuple of two callables: a work
    function (taking no arguments) and a callback function (taking a
    single argument, the return value of the work function).

    If called with 'asynchronous' true, the return value is an AsyncWorker
    object which can be used as described. If 'asynchronous' is false, the
    worker and callback functions are called synchronously and the
    return value of the callback is returned.
    """

    @wraps(func)
    def wrapper(*args, asynchronous=False, **kwargs):
        work, callback = func(*args, **kwargs)
        if asynchronous:
            return AsyncWorker(work, callback)
        else:
            return callback(work())
    return wrapper


class TomlFile(dict):
    """TOML file mapped as a dict."""

    def __init__(self, filename):
        super().__init__()
        if filename.exists():
            with open(filename, 'r') as f:
                self.update(toml.load(f), write=False)
        self.filename = filename

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.write()

    def update(self, other, write=True):
        super().update(other)
        if write:
            self.write()

    def write(self):
        with open(self.filename, 'w') as f:
            toml.dump(self, f)


class Config(TomlFile):
    """Geomaker config file mapped to a dict.
    Usually found at ~/.config/geomaker.toml.
    """

    def __init__(self):
        super().__init__(CONFIG_FILE)

    def verify(self, querier):
        if not 'email' in self:
            querier.message(
                'E-mail address',
                'An e-mail address must be configured to make requests to the Norwegian Mapping Authority',
            )
            self['email'] = querier.query_str('E-mail address', 'E-mail:')


class Data(TomlFile):
    """Geomaker data file mapped to a dict.
    Usually found at ~/.local/share/geomaker/geomaker.toml
    """

    def __init__(self):
        super().__init__(DATA_FILE)


class Polygon(DeclarativeBase):
    """ORM representation of a polygon."""

    __tablename__ = 'polygon'

    id = sql.Column(sql.Integer, primary_key=True)
    name = sql.Column(sql.String, nullable=False)

    # Sequence of points (counterclockwise order) making up this polygon
    # The last point must equal the first one.
    points = orm.relationship(
        'Point', order_by='Point.id', back_populates='polygon', lazy='immediate',
        cascade='save-update, merge, delete, delete-orphan',
    )

    # Thumbnail image for each project
    thumbnails = orm.relationship(
        'Thumbnail', back_populates='polygon', lazy='immediate',
        cascade='save-update, merge, delete, delete-orphan',
    )

    # GeoTIFF intermediate association
    assocs = orm.relationship(
        'PolyTIFF', back_populates='polygon', lazy='immediate',
        cascade='save-update, merge, delete, delete-orphan'
    )

    # Pending jobs for acquiring new GeoTIFF files
    jobs = orm.relationship(
        'Job', order_by='Job.jobid', back_populates='polygon', lazy='immediate',
        cascade='save-update, merge, delete, delete-orphan'
    )

    # The leaflet.js internal ID of this polygon. The lifetime of this
    # information is limited to one running of the program, and can be
    # different each time, so it is not persisted in the databse. It
    # is maintained as a bidirectional dict in the database object. We
    # provide a property-like interface to it from here.
    @property
    def lfid(self):
        return db.lfid.inverse.get(self.id, None)

    @lfid.setter
    def lfid(self, value):
        db.update_lfid(self, value)

    def geometry(self, coords='latlon'):
        """List of x, y coordinates (in a given coordinate system)
        making up this polygon.
        """
        for p in self.points:
            yield p.in_coords(coords)

    def edges(self, coords='latlon'):
        a, b = tee(self.geometry(coords), 2)
        for pta, ptb in zip(a, islice(b, 1, None)):
            yield (pta, ptb)

    @property
    def west(self):
        """Westernmost bounding point (longitude)."""
        return min(x for x,_ in self.geometry())

    @property
    def east(self):
        """Easternmost bounding point (longitude)."""
        return max(x for x,_ in self.geometry())

    @property
    def south(self):
        """Southernmost bounding point (latitude)."""
        return min(y for _,y in self.geometry())

    @property
    def north(self):
        """Northernmost bounding point (latitude)."""
        return max(y for _,y in self.geometry())

    @property
    def area(self):
        return geojson_area({'type': 'Polygon', 'coordinates': [list(self.geometry())]})

    @contextmanager
    def _assoc_query(self, cls, **kwargs):
        """Helper function for constructing a query for associated
        objects. 'Cls' should be the class of the resulting objects,
        followed by a keyword argument for each attribute to filter
        on.
        """
        filters = [cls.polygon == self]
        for key, value in kwargs.items():
            filters.append(getattr(cls, key) == value)
        with db.session() as s:
            yield s.query(cls).filter(*filters)

    def _single_assoc(self, cls, **kwargs):
        """Helper function for returning a single associated object,
        or None. Same interface as '_assoc_query'.
        """
        with self._assoc_query(cls, **kwargs) as q:
            return q.one_or_none()

    def thumbnail(self, project):
        """Get the thumbnail object associated with a project."""
        return self._single_assoc(Thumbnail, project=project)

    def dedicated(self, project):
        """Get the dedicated GeoTIFF file associated with a project."""
        obj = self._single_assoc(PolyTIFF, project=project, dedicated=True)
        if obj:
            obj = obj.geotiff
        return obj

    def ntiles(self, project):
        """Get the number of (non-dedicated) GeoTIFF files associated with a project."""
        with self._assoc_query(PolyTIFF, project=project, dedicated=False) as q:
            return q.count()

    def tiles(self, project):
        """Iterate over all (non-dedicated) GeoTIFF files assocaited with a project."""
        with self._assoc_query(PolyTIFF, project=project, dedicated=False) as q:
            for assoc in q:
                yield assoc.geotiff

    def delete_all_tiffs(self):
        """Remove all assocaited GeoTIFF files."""
        with self._assoc_query(PolyTIFF) as q:
            q.delete()

    def delete_dedicated(self, project):
        """Remove the dedicated GeoTIFF file associated with a project."""
        db.delete_if(self.dedicated(project))
        self.maybe_delete_thumbnail(project)

    def delete_tiles(self, project):
        """Remove the non-dedicated GeoTIFF files associated with a
        project. This may not actually delete the files from disk, if
        they are associated with another polygon.
        """
        with self._assoc_query(PolyTIFF, project=project, dedicated=False) as q:
            q.delete()
        self.maybe_delete_thumbnail(project)

    def njobs(self, **kwargs):
        """The number of jobs currently running matching the keyword argument filters."""
        with self._assoc_query(Job, **kwargs) as q:
            return q.count()

    def job(self, project, dedicated):
        """Get the currently running job for the given project."""
        return self._single_assoc(Job, project=project, dedicated=dedicated)

    def delete_job(self, project, dedicated):
        """Delete the currently running job for the given project."""
        obj = self.job(project, dedicated)
        db.delete_if(obj)

    def create_job(self, project, dedicated, email):
        """Create a new job. This will fail if existing data files or
        jobs exist matching the given criteria.
        """
        if dedicated:
            assert self.dedicated(project) is None
        else:
            assert self.ntiles(project) == 0
        assert self.job(project, dedicated) is None

        coords = list(self.geometry('utm33n'))
        coords = [(int(x), int(y)) for x, y in coords]
        coords = ';'.join(f'{x},{y}' for x, y in coords)
        params = {
            'CopyEmail': email,
            'Projects': project,
            'CoordInput': coords,
            'ProjectMerge': 1 if dedicated else 0,
            'InputWkid': 25833,      # ETRS89 / UTM zone 33N
            'Format': 5,             # GeoTIFF,
            'NHM': 1,                # National altitude models
        }

        code, response = make_request('startExport', params)
        if response is None:
            return f'HTTP code {code}'
        elif 'Error'in response:
            return response['Error']
        elif not response.get('Success', False):
            return 'Unknown error'

        job = Job(polygon=self, project=project, dedicated=dedicated, jobid=response['JobID'])
        with db.session() as s:
            s.add(job)

    def geographic_angle(self, coords):
        points = list(self.geometry())
        center = sum(points) / len(points)
        up = convert_latlon(center + np.array([0.1, 0.0]), coords)
        vec = up - convert_latlon(center, coords)
        return -np.arctan2(vec[1], vec[0])

    def _rectangularize(self, mode, rotate, coords):
        points = list(self.geometry(coords))
        if rotate == 'free' and mode == 'interior':
            rect, area, theta = polyfit.largest_rectangle(points)
        elif rotate == 'north' and mode == 'interior':
            theta = self.geographic_angle(coords)
            rect, area = polyfit.largest_rotated_rectangle(points, theta)
        elif mode == 'interior':
            rect, area = polyfit.largest_aligned_rectangle(points)
            theta = 0.0
        elif rotate == 'free':
            rect, area, theta = polyfit.smallest_rectangle(points)
        elif rotate == 'north':
            theta = self.geographic_angle(coords)
            rect, area = polyfit.smallest_rotated_rectangle(points, theta)
        else:
            rect, area = polyfit.smallest_aligned_rectangle(points)
            theta = 0.0
        return rect, area, theta

    def check_area(self, mode, rotate, coords):
        reference_area = polyfit.polygon_area(list(self.geometry(coords)))
        _, actual_area, theta = self._rectangularize(mode, rotate, coords)
        return abs(actual_area - reference_area) / reference_area, theta

    def generate_meshgrid(self, mode, rotate, coords, resolution=None, maxpts=None):
        rect, _, _ = self._rectangularize(mode, rotate, coords)
        a, b, c, d, *_ = rect
        width = np.linalg.norm(b - a)
        height = np.linalg.norm(c - b)
        if resolution is None:
            resolution = max(width, height) / maxpts
        nx = int(ceil(height / resolution))
        ny = int(ceil(width / resolution))
        xt, yt = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing='ij')
        x = (1-xt) * (1-yt) * d[1] + xt * (1-yt) * a[1] + xt * yt * b[1] + (1-xt) * yt * c[1]
        y = (1-xt) * (1-yt) * d[0] + xt * (1-yt) * a[0] + xt * yt * b[0] + (1-xt) * yt * c[0]
        return x, y

    def interpolate(self, project, x, y):
        assert x.shape == y.shape
        data = np.zeros(x.shape)
        if self.dedicated(project):
            tiffs = [self.dedicated(project)]
        else:
            tiffs = list(self.tiles(project))
        for tiff in tiffs:
            tiff.interpolate(data, x, y)
        return data

    def maybe_delete_thumbnail(self, project):
        if self.dedicated(project) or self.ntiles(project) > 0:
            return
        db.delete_if(self.thumbnail(project))

    def update_thumbnail(self, project, dedicated):
        """Update the thumbnail for the given project.
        If a thumbnail already exists for that project, this function
        will silently do nothing, unless 'dedicated' is true.
        The source for the thumbnail will come from the dedicated data
        file, if one exists, or the tiled data files if it does not.
        """
        if self.thumbnail(project) is not None and not dedicated:
            return
        db.delete_if(self.thumbnail(project))

        x, y = self.generate_meshgrid('exterior', 'none', 'utm33n', maxpts=640)
        data = self.interpolate(project, x, y)
        filename = THUMBNAIL_ROOT / (hashlib.sha256(data.data).hexdigest() + '.png')
        image.array_to_image(data, 'terrain', True, filename)

        # Create a new Thumbnail object in the database
        thumb = Thumbnail(filename=str(filename), project=project, polygon=self)
        with db.session() as s:
            s.add(thumb)


class Thumbnail(DeclarativeBase):
    """ORM representation of a thumbnail image associated with a
    polygon and a project.
    """

    __tablename__ = 'thumbnail'

    id = sql.Column(sql.Integer, primary_key=True)
    filename = sql.Column(sql.String, nullable=False)
    project = sql.Column(sql.String, nullable=False)
    polygon_id = sql.Column(sql.Integer, sql.ForeignKey('polygon.id'), nullable=False)
    polygon = orm.relationship('Polygon', back_populates='thumbnails', lazy='immediate')

@sql.event.listens_for(Thumbnail, 'after_delete')
def delete_thumbnail(mapper, connection, thumbnail):
    Path(thumbnail.filename).unlink()


class Point(DeclarativeBase):
    """A point in latitude and longitude coordinates."""

    __tablename__ = 'point'

    id = sql.Column(sql.Integer, primary_key=True)
    x = sql.Column(sql.Float, nullable=False)
    y = sql.Column(sql.Float, nullable=False)
    polygon_id = sql.Column(sql.Integer, sql.ForeignKey('polygon.id'), nullable=False)
    polygon = orm.relationship('Polygon', back_populates='points', lazy='immediate')

    def in_coords(self, coords):
        return convert_latlon(np.array([self.x, self.y]), coords)


class GeoTIFF(DeclarativeBase):
    """ORM representation of a GeoTIFF file."""

    __tablename__ = 'geotiff'

    id = sql.Column(sql.Integer, primary_key=True)
    filename = sql.Column(sql.String, nullable=False)
    east = sql.Column(sql.Float, nullable=False)
    west = sql.Column(sql.Float, nullable=False)
    south = sql.Column(sql.Float, nullable=False)
    north = sql.Column(sql.Float, nullable=False)

    # Polygon intermediate association
    assocs = orm.relationship(
        'PolyTIFF', back_populates='geotiff', lazy='immediate',
        cascade='save-update, merge, delete, delete-orphan',
    )

    @lru_cache(maxsize=1)
    def _dataset(self):
        """Cached TiffFile object."""
        return tif.TiffFile(str(self.filename))

    @lru_cache(maxsize=1)
    def as_array(self):
        """Cached data array object."""
        return self._dataset().asarray()

    @property
    def shape(self):
        return self.as_array().shape

    @property
    def resolution(self):
        rx, ry, _ = self._dataset().geotiff_metadata['ModelPixelScale']
        return rx, ry

    def populate(self):
        """Refresh the bounding box data."""
        rx, ry = self.resolution
        nx, ny = self.shape
        i, j, k, x, y, z = self._dataset().geotiff_metadata['ModelTiepoint']

        assert i == j == k == z == 0

        # Compute bounding box
        self.west = x + 0.5 * rx
        self.east = self.west + rx * (ny - 1)
        self.north = y - 0.5 * ry
        self.south = self.north - ry * (nx - 1)

    def interpolate(self, data, x, y):
        """Interpolate onto a data array. 'Data', 'x' and 'y' should
        be 2D arrays with the same shape. For all points where x and y
        fit in the bounding box of this GeoTIFF file, the elements of
        'data' are modified by bilinear interpolation.

        The values of 'data' are only increased, never decreased. This
        to facilitate several overlapping GeoTIFF files where missing
        height values may be encoded as large negative values. For
        best results, the data array should be initialized with zeros
        or suitably negative values.
        """
        rx, ry = self.resolution

        # Mask of which indices apply to this TIFF
        I, J = np.where((self.south <= x) & (x < self.north) & (self.west <= y) & (y < self.east))
        x = (self.north - x[I, J]) / rx
        y = (y[I, J] - self.west) / ry

        # Compute indices of the element for each point
        left = np.floor(x).astype(int)
        down = np.floor(y).astype(int)

        # Reference coordinates for each point
        ref_left = x - left
        ref_down = y - down

        # Interpolate
        refdata = self.as_array()
        refdata[np.where(refdata < 0)] = 0

        data[I, J] = np.maximum(
            data[I, J],
            refdata[left,   down]   * (1 - ref_left) * (1 - ref_down) +
            refdata[left+1, down]   * ref_left       * (1 - ref_down) +
            refdata[left,   down+1] * (1 - ref_left) * ref_down +
            refdata[left+1, down+1] * ref_left       * ref_down
        )

@sql.event.listens_for(GeoTIFF, 'after_delete')
def delete_geotiff(mapper, connection, geotiff):
    Path(geotiff.filename).unlink()


class PolyTIFF(DeclarativeBase):
    """ORM intermediate association table between Polygons and GeoTIFF objects."""

    __tablename__ = 'polytiff'

    polygon_id = sql.Column(sql.Integer, sql.ForeignKey('polygon.id'), primary_key=True)
    geotiff_id = sql.Column(sql.Integer, sql.ForeignKey('geotiff.id'), primary_key=True)
    dedicated = sql.Column(sql.Boolean, nullable=False)
    project = sql.Column(sql.String, nullable=False)

    polygon = orm.relationship('Polygon', back_populates='assocs', lazy='immediate')
    geotiff = orm.relationship('GeoTIFF', back_populates='assocs', lazy='immediate')


class Job(DeclarativeBase):
    """ORM representation of a pending external job."""

    __tablename__ = 'job'

    id = sql.Column(sql.Integer, primary_key=True)
    polygon_id = sql.Column(sql.Integer, sql.ForeignKey('polygon.id'))
    project = sql.Column(sql.String, nullable=False)
    dedicated = sql.Column(sql.Boolean, nullable=False)
    jobid = sql.Column(sql.Integer, nullable=False)
    stage = sql.Column(sql.String, nullable=False, default='new')
    error = sql.Column(sql.String, nullable=True)
    url = sql.Column(sql.String, nullable=True)

    polygon = orm.relationship('Polygon', back_populates='jobs', lazy='immediate')

    @asynchronous
    def refresh(self):
        """Async-capable method for refreshing the status of the job."""
        worker = partial(make_request, 'exportStatus', {'JobID': self.jobid})
        callback = self.refresh_commit
        return worker, callback

    # Used as a callback for refresh()
    # SQLAlchemy doesn't like it when you modify objects outside their original thread
    def refresh_commit(self, args):
        if self.stage == 'downloaded':
            return
        code, response = args
        if response is None:
            self.stage = 'error'
            self.error = f'HTTP code {code}'
        else:
            self.stage = response['Status']
            if self.stage == 'complete':
                self.url = response['Url']
        db.commit()
        return (self.polygon, self.project)

    @asynchronous
    def download(self):
        """Async-capable method for downloading job data files.
        This automatically updates the polygon thumbnail and finally
        deletes the job.
        """
        assert self.stage == 'complete'
        assert self.url is not None
        worker = partial(download_geotiffs, self.url, self.project, self.dedicated)
        callback = self.download_commit
        return worker, callback

    # Used as a callback for download()
    # SQLAlchemy doesn't like it when you modify objects outside their original thread
    def download_commit(self, filenames):
        for filename in filenames:
            geotiff = GeoTIFF(filename=str(filename))
            geotiff.populate()
            polytiff = PolyTIFF(polygon=self.polygon, geotiff=geotiff, project=self.project, dedicated=self.dedicated)
            with db.session() as s:
                s.add(geotiff)
                s.add(polytiff)

        # TODO: Generate thumbnails asynchronously
        self.polygon.update_thumbnail(self.project, self.dedicated)
        retval = (self.polygon, self.project)
        with db.session() as s:
            s.delete(self)
        return retval


class Database:
    """Primary database interface for use by the GUI. Intended to be
    used as a singleton.
    """

    def __init__(self):
        self.engine = sql.create_engine(f'sqlite:///{DATA_ROOT}/db.sqlite3')
        DeclarativeBase.metadata.create_all(self.engine)

        # The session object lasts for the lifetime of the program
        self._session = orm.sessionmaker(bind=self.engine)()

        # A bidirectional dictionary mapping database Polygon IDs to leaflet.js IDs
        self.lfid = bidict()

        # A list of objects to be notified by changes
        self.listeners = []

    @contextmanager
    def session(self):
        """A context manager yielding a session object, committing
        after use or rolling back on error.
        """
        session = self._session
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise

    def commit(self):
        """Forcibly commit all pending changes."""
        with self.session() as s:
            pass

    def __iter__(self):
        """Iterate over polygons in sorted order by name."""
        with self.session() as s:
            yield from s.query(Polygon).order_by(Polygon.name)

    def __len__(self):
        with self.session() as s:
            return s.query(Polygon).count()

    def __getitem__(self, index):
        """Get the nth polygon by order of name."""
        with self.session() as s:
            return s.query(Polygon).order_by(Polygon.name)[index]

    @contextmanager
    def _job_query(self, stage=None):
        """Helper function for querying jobs."""
        with self.session() as s:
            q = s.query(Job)
            if stage is not None:
                q = q.filter(Job.stage == stage)
            yield q

    def njobs(self, stage=None):
        """Return the number of pending external jobs."""
        with self._job_query(stage) as q:
            return q.count()

    def jobs(self, stage=None):
        """Iterate over pending external jobs."""
        with self._job_query(stage) as q:
            yield from q.order_by(Job.jobid)

    def delete_if(self, obj):
        """Delete obj if it is not None."""
        if obj is not None:
            with self.session() as s:
                s.delete(obj)

    def poly_by_lfid(self, lfid):
        """Look up a polygon by its leaflet.js internal ID."""
        with self.session() as s:
            return s.query(Polygon).get(self.lfid[lfid])

    def index(self, poly=None, lfid=None):
        """Get the current index (in sorted order by name) of a polygon,
        possibly given by its leaflet.js internal ID.
        """
        if poly is None:
            poly = self.poly_by_lfid(lfid)
        with self.session() as s:
            return s.query(Polygon).filter(Polygon.name < poly.name).count()

    def update_lfid(self, poly, lfid):
        """Update the leaflet.js internal ID of the polygon 'poly' to 'lfid'.
        This includes potentially deleting an existing lfid <-> poly binding.
        """
        if poly.id in self.lfid.inverse:
            del self.lfid.inverse[poly.id]
        if lfid is not None:
            self.lfid[lfid] = poly.id

    def notify(self, obj):
        """Register 'obj' as a receiver of events."""
        self.listeners.append(obj)

    def message(self, method, *args):
        """Send the message 'method' to each listener."""
        caller = methodcaller(method, *args)
        for listener in self.listeners:
            caller(listener)

    def update_name(self, index, name):
        """Change the name of the nth polygon."""
        poly = self[index]

        self.message('before_reset', poly.lfid)
        poly.name = name
        self.commit()
        self.message('after_reset')

    def update_points(self, lfid, data):
        """Change the points of a given polygon by leaflet.js internal ID.
        The 'data' argument is a GeoJSON object in string form.
        """
        points = json.loads(data)['geometry']['coordinates'][0]
        poly = self.poly_by_lfid(lfid)
        with self.session() as s:
            for point in poly.points:
                s.delete(point)
            for x, y in points:
                s.add(Point(x=x, y=y, polygon=poly))
            poly.delete_all_tiffs()
            poly.thumbnail = None

    def create(self, lfid, name, data):
        """Create a new polygon with a given name and leaflet.js internal ID.
        The 'data' argument is a GeoJSON object in string form.
        """
        points = json.loads(data)['geometry']['coordinates'][0]

        poly = Polygon(name=name)
        for x, y in points:
            Point(x=x, y=y, polygon=poly)
        with self.session() as s:
            s.add(poly)
        poly.lfid = lfid

        index = self.index(poly=poly)
        self.message('before_insert', index)
        self.message('after_insert')

    def delete(self, lfid):
        """Delete a polygon by its leaflet.js internal ID."""
        poly = self.poly_by_lfid(lfid)
        index = self.index(poly=poly)

        self.message('before_delete', index)
        with self.session() as s:
            s.delete(poly)
        self.message('after_delete')


db = Database()
