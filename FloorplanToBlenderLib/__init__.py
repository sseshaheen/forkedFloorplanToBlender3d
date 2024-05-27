"""
FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg

This file create python package

Example usage of package :

from FloorplanToBlenderLib import * # floorplan to blender lib

detect...
generate...
IO...
transform...
dialog...
execution...

"""
import logging
import config
from .globalConfig import LOGGING_VERBOSE

# Configure logging
if LOGGING_VERBOSE:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

__all__ = [
    "image",
    "detect",
    "generate",
    "IO",
    "transform",
    "dialog",
    "execution",
    "const",
    "generator",
    "draw",
    "calculate",
    "config",
    "stacking",
    "floorplan",
]
