from contextlib import contextmanager
from itertools import chain

import toml
import xdg

from .util import SingletonMeta



DATA_ROOT = xdg.XDG_DATA_HOME / 'geomaker'
DATA_FILE = DATA_ROOT / 'geomaker.toml'
THUMBNAIL_ROOT = DATA_ROOT / 'thumbnails'
CONFIG_FILE = xdg.XDG_CONFIG_HOME / 'geomaker.toml'


def create_directories(projects):
    directories = chain(
        [DATA_ROOT, THUMBNAIL_ROOT],
        (DATA_ROOT / proj for proj in projects)
    )
    for directory in directories:
        if not directory.exists():
            directory.mkdir()


def project_file(project, filename):
    return DATA_ROOT / project / filename


def thumbnail_file(filename):
    return THUMBNAIL_ROOT / filename


class TomlFile(dict, metaclass=SingletonMeta):
    """Singleton TOML file mapped as a dict."""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._suspended = False

        if filename.exists():
            with open(filename, 'r') as f:
                self.update(toml.load(f), write=False)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.write()

    def update(self, other, write=True):
        super().update(other)
        if write:
            self.write()

    def write(self, force=False):
        if force or not self._suspended:
            with open(self.filename, 'w') as f:
                toml.dump(dict(self), f)

    @contextmanager
    def suspend_write(self):
        prev_val = self._suspended
        self._suspended = True
        yield self
        self._suspended = prev_val
        self.write()


class ConfigFile(TomlFile):
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


class DataFile(TomlFile):
    """Geomaker data file mapped to a dict.
    Usually found at ~/.local/share/geomaker/geomaker.toml
    """

    def __init__(self):
        super().__init__(DATA_FILE)
