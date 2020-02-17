#!/usr/bin/env python3

from distutils.core import setup
from setuptools import find_packages

setup(
    name='GeoMaker',
    version='0.1',
    description='Height data extraction tool',
    author='Eivind Fonn',
    author_email='evfonn@gmail.com',
    license='GPL3',
    url='https://github.com/TheBB/geomaker',
    packages=find_packages(),
    package_data={
        'geomaker': [
            'assets/map.html', 'assets/map.js',
            'alembic.ini', 'migrations/env.py', 'migrations/versions/*.py',
        ],
    },
    entry_points={
        'console_scripts': ['geomaker=geomaker.__main__:main'],
    },
    extras_require={
        'Splines': ['Splipy'],
        'STL': ['numpy-stl'],
        'VTK': ['vtk'],
    },
    install_requires=[
        'alembic',
        'area',
        'bidict',
        'indexed.py',
        'humanfriendly',
        'matplotlib',
        'numpy',
        'Pillow',
        'pygeotile',
        'PyQt5',
        'PyQtWebEngine',
        'requests',
        'scipy',
        'SQLAlchemy',
        'tifffile',
        'toml',
        'utm',
        'xdg',
    ],
)
