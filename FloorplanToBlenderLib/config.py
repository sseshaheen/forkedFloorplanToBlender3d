import configparser
import os
import cv2
import json

from . import IO
from . import const
from . import calculate

"""
Config
This file contains functions for handling config files.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""

# TODO: settings for coloring all objects
# TODO: add config security check, before start up!
# TODO: safe read, use this func instead of repeating code everywhere!
# TODO: add blender path addition to system.ini

def read_calibration(floorplan):
    """
    Read all calibrations.
    @Param floorplan: Floorplan object to read calibration for.
    @Return: Wall size calibration value.
    """
    if floorplan.wall_size_calibration == 0:
        floorplan.wall_size_calibration = create_image_scale_calibration(floorplan)
    return floorplan.wall_size_calibration


def create_image_scale_calibration(floorplan, got_settings=False):
    """
    Create and save image size calibrations.
    @Param floorplan: Floorplan object to create calibration for.
    @Param got_settings: Boolean to indicate if settings were retrieved.
    @Return: Wall width average.
    """
    calibration_img = cv2.imread(floorplan.calibration_image_path)
    return calculate.wall_width_average(calibration_img)

def generate_file():
    """
    Generate a new config file if it does not exist.
    """
    # Create system settings
    conf = configparser.ConfigParser()
    conf["SYSTEM"] = {
        const.STR_OVERWRITE_DATA: const.DEFAULT_OVERWRITE_DATA,  # TODO: implement!
        const.STR_BLENDER_INSTALL_PATH: IO.get_blender_os_path(),
        const.STR_OUT_FORMAT: json.dumps(const.DEFAULT_OUT_FORMAT),
    }

    os.makedirs(os.path.dirname(const.SYSTEM_CONFIG_FILE_NAME), exist_ok=True)
    with open(const.SYSTEM_CONFIG_FILE_NAME, "w") as configfile:
        conf.write(configfile)

    # Create default floorplan settings
    conf = configparser.ConfigParser()
    conf["IMAGE"] = {
        const.STR_IMAGE_PATH: json.dumps(const.DEFAULT_IMAGE_PATH),
        "COLOR": json.dumps([0, 0, 0]),
    }

    conf["TRANSFORM"] = {
        "position": json.dumps([0, 0, 0]),
        "rotation": json.dumps([0, 0, 90]),
        "scale": json.dumps([1, 1, 1]),
        "margin": json.dumps([0, 0, 0]),
    }

    conf[const.FEATURES] = {
        const.STR_FLOORS: json.dumps(const.DEFAULT_FEATURES),
        const.STR_ROOMS: json.dumps(const.DEFAULT_FEATURES),
        const.STR_WALLS: json.dumps(const.DEFAULT_FEATURES),
        const.STR_DOORS: json.dumps(const.DEFAULT_FEATURES),
        const.STR_WINDOWS: json.dumps(const.DEFAULT_FEATURES),
    }

    conf[const.SETTINGS] = {
        const.STR_REMOVE_NOISE: json.dumps(const.DEFAULT_REMOVE_NOISE),
        const.STR_RESCALE_IMAGE: json.dumps(const.DEFAULT_RESCALE_IMAGE),
    }

    conf[const.WALL_CALIBRATION] = {
        const.STR_CALIBRATION_IMAGE_PATH: json.dumps(
            const.DEFAULT_CALIBRATION_IMAGE_PATH
        ),
        const.STR_WALL_SIZE_CALIBRATION: json.dumps(
            const.DEFAULT_WALL_SIZE_CALIBRATION
        ),
    }

    with open(const.IMAGE_DEFAULT_CONFIG_FILE_NAME, "w") as configfile:
        conf.write(configfile)


def show(conf):
    """
    Visualize all config settings.
    @Param conf: Config object to visualize.
    """
    for key in conf:
        print(key, conf[key])


def update(path, key, config):
    """
    Update a config category with a config object.
    @Param path: Path to the config file.
    @Param key: Config category key.
    @Param config: Config object to update.
    """
    conf = get_all(path)
    conf[key] = config
    with open(path, "w") as configfile:
        conf.write(configfile)


def file_exist(name):
    """
    Check if a file exists.
    @Param name: Name of the file.
    @Return: Boolean indicating if the file exists.
    """
    return os.path.isfile(name)


def get_all(path):
    """
    Read and return all config values.
    @Param path: Path to the config file.
    @Return: Config object with all values.
    """
    return get(path)


def get(config_path, *args):
    """
    Read and return values from a config file.
    @Param config_path: Path to the config file.
    @Return: Config object with requested values.
    """
    conf = configparser.ConfigParser()

    if not file_exist(config_path):
        generate_file()
    conf.read(config_path)

    for key in args:
        conf = conf[key]

    if args is None:
        return conf
    else:
        return conf


def get_default_image_path():
    """
    Get the default image path from the config file.
    @Return: Default image path.
    """
    return get(const.IMAGE_DEFAULT_CONFIG_FILE_NAME, "IMAGE", const.STR_IMAGE_PATH)


def get_default_blender_installation_path():
    """
    Get the default Blender installation path from the config file.
    @Return: Default Blender installation path.
    """
    return get(const.SYSTEM_CONFIG_FILE_NAME, "SYSTEM", const.STR_BLENDER_INSTALL_PATH)
