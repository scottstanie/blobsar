# -*- coding: utf-8 -*-
"""
    Setup file for blobsar.
    Use setup.cfg to configure your project.
"""
import sys
from pkg_resources import VersionConflict, require
from setuptools import setup

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup()
