# Geomaker

Geomaker is a map-based GUI for extracting detailed map data from
regions in Norway based on the web services of the Norwegian Mapping
Authority.

## Installation

In simplest terms:

```bash
git clone https://github.com/TheBB/geomaker
cd geomaker
pip install --user .
```

This should install Geomaker and all its dependencies *locally*,
typically to your `~/.local` folder. The executable is called
`geomaker`, generally located in `~/.local/bin/geomaker`.

There are a number of ways this might fail:

- If `pip` is not installed, install it.
- If `pip` points to a Python 2 installation, your distribution's
  Python 3 `pip` might be called `pip3`.
- If PyQt5 or some other dependencies fail to install, you may want to
  install them manually using your system's package manager. See the
  detailed list of dependencies below. After doing this, you can
  install geomaker by using `--no-deps` with `pip`.
- A recent version of `pip` is highly recommended. Some PyQt5 packages
  may fail to install with older versions. To update `pip` itself, use
  `pip install --user --upgrade pip`. Note that this will install
  `pip` locally, and it may shadow the system-installed version of
  `pip` depending on your `PATH` settings, but it shouldn't overwrite
  it.
  
If you wish to frequently update Geomaker, I recommend also to install
with the `--editable` flag, so that the installed version of Geomaker
automatically tracks the repository contents.

## Dependencies

The following packages are simple Python-only packages:

- area (for calculating areas in spherical coordinate systems)
- bidict (bidirectional dictionaries)
- matplotlib (for color maps)
- numpy (no scientific code can be without)
- requests (for making API queries to hoydedata.no)
- sqlalchemy (for maintaining persistent data on disk)
- tifffile (for reading GeoTIFF files)
- toml (for the config file)
- utm (for converting to and from UTM coordinates)
- xdg (for accessing XDG paths)

In addition, the PyQt5 bindings are necessary:

- PyQt5
- PyQtWebEngine