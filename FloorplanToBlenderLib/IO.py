import json
import os
from shutil import which
import shutil
import cv2
import platform
from sys import platform as pf
import numpy as np
import logging
from . import const
from . import image
from . import config
from .globalConf import load_config_from_json, LOGGING_VERBOSE, DEBUG_MODE

"""
IO
This file contains functions for handling files.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""
def configure_logging():
    if LOGGING_VERBOSE:
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')
        
        log_dir_path = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'])
        log_file_path = os.path.join(log_dir_path, 'debug.log')
        os.makedirs(log_dir_path, exist_ok=True)

        # Create a logger
        logger = logging.getLogger('debug_logger')
        logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

        # Create a file handler to log everything
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler to log warnings and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set the root logger's level to WARNING to avoid interference
        logging.getLogger().setLevel(logging.WARNING)

        return logger
    return None





def save_debug_info(filename, data):
    """
    Save debug information to a file if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')

        DEBUG_STORAGE_PATH = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'], 'txt')
        if not os.path.exists(DEBUG_STORAGE_PATH):
            os.makedirs(DEBUG_STORAGE_PATH)
        filepath = os.path.join(DEBUG_STORAGE_PATH, filename)
        with open(filepath, 'a') as file:
            file.write(str(data))
        # if LOGGING_VERBOSE:
            # logger.debug(f'Saved debug info: {filepath}')


def find_reuseable_data(image_path, path):
    """
    Checks if floorplan data already exists and can be reused.
    @Param image_path: Path to the image.
    @Param path: Path to search for reusable data.
    @Return: Path to reusable image data, shape of the reusable data, else None.
    """
    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            try:
                with open(os.path.join(path, dir, const.TRANSFORM_PATH)) as f:
                    data = f.read()
                js = json.loads(data)
                if image_path == js[const.STR_IMAGE_PATH]:
                    if LOGGING_VERBOSE:
                        logger.debug(f'Reusable data found for image: {image_path}')
                    return js[const.STR_ORIGIN_PATH], js[const.STR_SHAPE]
            except IOError:
                continue
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'No reusable data found for image: {image_path}')
    return None, None


def find_files(filename, search_path):
    """
    Find filename in root search path.
    @Param filename: Name of the file to search for.
    @Param search_path: Root directory to start the search.
    @Return: Full path to the found file, else None.
    """
    for root, _, files in os.walk(search_path):
        if filename in files:
            if LOGGING_VERBOSE:
                logger = configure_logging()
                if logger:
                    logger.debug(f'File found: {filename} at {root}')
            return os.path.join(root, filename)
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'File not found: {filename}')
    return None


def blender_installed():
    """
    Find path to blender installation.
    Might be error prone, tested on Ubuntu and Windows.
    @Return: Path to Blender executable if found, else None.
    """
    if pf == "linux" or pf == "linux2":
        # Linux
        return find_files("blender", "/")
    elif pf == "darwin":
        # OS X
        return find_files("blender", "/")  # TODO: this needs to be tested!
    elif pf == "win32":
        # Windows
        return find_files("blender.exe", "C:\\")

def get_blender_os_path():
    """
    Get the default installation path of Blender based on the operating system.
    @Return: Default Blender installation path for the current OS.
    """
    _platform = platform.system()
    if _platform.lower() in ["linux", "linux2", "ubuntu"]:
        return const.LINUX_DEFAULT_BLENDER_INSTALL_PATH
    elif _platform.lower() == "darwin":
        return const.MAC_DEFAULT_BLENDER_INSTALL_PATH
    elif "win" in _platform.lower():
        return const.WIN_DEFAULT_BLENDER_INSTALL_PATH


def read_image(path, floorplan=None):
    """
    Read image, resize/rescale and return with grayscale.
    @Param path: Path to the image file.
    @Param floorplan: Floorplan object containing image processing parameters.
    @Return: Original image, Grayscale image, Scale factor.
    """
    # Read floorplan image
    img = cv2.imread(path)
    if img is None:
        logger.error(f"ERROR: Image {path} could not be read by OpenCV library.")
        raise IOError

    scale_factor = 1
    if floorplan is not None:
        if floorplan.remove_noise:
            img = image.denoising(img, caller='read_image')
        if floorplan.rescale_image:
            calibrations = config.read_calibration(floorplan)
            floorplan.wall_size_calibration = calibrations  # Store for debug
            if calibrations is None:
                logger.error("ERROR: Calibration data is None. Using default scale factor.")
                scale_factor = 1
            else:
                scale_factor = image.detect_wall_rescale(float(calibrations), img)
                if scale_factor is None:
                    logger.warning(
                        "WARNING: Auto rescale failed due to non-good walls found in image."
                        + "If rescale still is needed, please rescale manually."
                    )
                    scale_factor = 1
                else:
                    img = image.cv2_rescale_image(img, scale_factor)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Read and processed image: {path}')
    save_debug_info('read_image.txt', {'path': path, 'scale_factor': scale_factor})

    return img, gray, scale_factor

def readlines_file(path):
    res = []
    """
    Read all lines from a file.
    @Param path: Path to the file.
    @Return: List of lines read from the file.
    """
    with open(path, "r") as f:
        res = f.readlines()
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Read lines from file: {path}')
    save_debug_info('readlines_file.txt', {'path': path, 'lines': res})
    return res


def ndarrayJsonDumps(obj):
    """
    Convert numpy array to JSON serializable format.
    @Param obj: Numpy object to be converted.
    @Return: List if the object is an ndarray, item if it is another numpy type.
    """
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError("Unknown type:", type(obj))


def save_to_file(file_path, data, show=True):
    """
    Save data to a file in JSON format.
    @Param file_path: Path to the output file.
    @Param data: Data to write to the file.
    @Param show: Boolean indicating if a success message should be printed.
    """
    full_path = file_path + const.SAVE_DATA_FORMAT
    with open(full_path, "w") as f:
        try:
            f.write(json.dumps(data, default=ndarrayJsonDumps))
        except TypeError as e:
            logger.error(f'Error saving data to file: {e}')
            raise
    if show:
        logger.info(f'Created file: {full_path}')
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Saved data to file: {full_path}')
    save_debug_info(f'save_to_file-{os.path.basename(file_path)}.txt', {'file_path': full_path, 'data': data})

def read_from_file(file_path):
    """
    Read data from a file in JSON format.
    @Param file_path: Path to the file.
    @Return: Data read from the file.
    """
    full_path = file_path + const.SAVE_DATA_FORMAT
    with open(full_path, "r") as f:
        data = json.loads(f.read())
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Read data from file: {full_path}')
    save_debug_info('read_from_file.txt', {'file_path': full_path, 'data': data})
    return data


def clean_data_folder(folder):
    """
    Remove old data files to avoid filling memory.
    @Param folder: Path to the data folder.
    """
    for root, dirs, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Cleaned data folder: {folder}')
    # save_debug_info('clean_data_folder.txt', {'folder': folder})

def create_new_floorplan_path(path):
    """
    Create the next free name for floorplan data.
    @Param path: Path to the floorplan directory.
    @Return: New path for the floorplan data.
    """
    res = 0
    for _, dirs, _ in os.walk(path):
        for _ in dirs:
            try:
                name_not_found = True
                while name_not_found:
                    if not os.path.exists(os.path.join(path, str(res))):
                        break
                    res += 1
            except Exception:
                continue

    res_path = os.path.join(path, str(res))
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Created new floorplan path: {res_path}')
    # save_debug_info('create_new_floorplan_path.txt', {'path': res_path})
    return res_path


def get_current_path():
    """
    Get the current working directory path.
    @Return: Path to the working directory.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Current path: {dir_path}')
    save_debug_info('get_current_path.txt', {'path': dir_path})
    return dir_path


def find_program_path(name):
    """
    Find the path of a program.
    @Param name: Name of the program to find.
    @Return: Path to the program if found, else None.
    """
    program_path = which(name)
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Program path for {name}: {program_path}')
    save_debug_info('find_program_path.txt', {'name': name, 'path': program_path})
    return program_path

def get_next_target_base_name(target_base, target_path):
    """
    Generate the next appropriate target name.
    If the Blender target file already exists, get the next ID.
    @Param target_base: Base name for the target.
    @Param target_path: Path to the target directory.
    @Return: Next target base name.
    """
    fid = 0
    if os.path.isfile("." + target_path):
        for file in os.listdir("." + const.TARGET_PATH):
            filename = os.fsdecode(file)
            if filename.endswith(const.BASE_FORMAT):
                fid += 1
        target_base += str(fid)

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f'Next target base name: {target_base}')
    save_debug_info('get_next_target_base_name.txt', {'target_base': target_base, 'target_path': target_path, 'fid': fid})
    return target_base
