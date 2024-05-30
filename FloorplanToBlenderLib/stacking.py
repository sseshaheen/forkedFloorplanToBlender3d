import logging
import numpy as np
from . import IO
from . import execution
from . import const
from . import floorplan
from . import transform
from .globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE
import os

# Configure logging
if LOGGING_VERBOSE:
    # Load the DEBUG_SESSION_ID from the JSON file
    debug_config = load_config_from_json('./config.json')
    
    log_dir_path = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'])
    log_file_path = os.path.join(log_dir_path, 'debug.log')
    os.makedirs(os.path.dirname(log_dir_path), exist_ok=True)



    # Create a logger
    logger = logging.getLogger(__name__)
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
        if LOGGING_VERBOSE:
            logger.debug(f'Saved debug info: {filepath}')

"""
Stacking
This file contains functions for handling stacking file parsing and creating larger stacking.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""

def parse_stacking_file(path):
    """
    Parse strictly formatted stacking files.
    These are used to more easily place many floorplans in one scene.
    @Param path: Path to the stacking file.
    @Return: List of worlds containing floorplans.
    """
    array_of_commands = IO.readlines_file(path)

    world = []
    worlds = []

    logger.info(f'Building stack from file {path}')
    if LOGGING_VERBOSE:
        logger.debug(f'Reading stacking file from: {path}')

    for index, line in enumerate(array_of_commands):
        args = line.split(" ")
        command = args[0]

        if command[0] in ["#", "\n", "", " "]:  # Ignore commented lines
            continue

        try:
            args.remove("\n")
        except Exception:
            pass

        new_args = []
        for cmd in args:
            if cmd == '"_"':
                new_args.append("None")
            else:
                new_args.append(cmd)
        args = new_args

        argstring = ""
        for index, arg in enumerate(args[1:]):
            if index == len(args[1:]) - 1:
                argstring += arg
            else:
                argstring += arg + ","

        logger.info(f'>Line {index} Command: {command}({argstring})')

        if command == "SEPARATE":
            worlds.append(world)
            world = []
        elif command == "CLEAR":
            eval(command + "(" + argstring + ")")
        else:
            world.extend(eval(command + "(" + argstring + ")"))

    worlds.extend(world)
    
    if LOGGING_VERBOSE:
        logger.debug(f'Parsed stacking file: {path}')
    save_debug_info('parse_stacking_file.txt', {'path': path, 'worlds': worlds})

    return worlds


def CLEAR():
    """
    Clear the data folder.
    """
    IO.clean_data_folder(const.BASE_PATH)
    logger.info('Cleared data folder.')
    if LOGGING_VERBOSE:
        logger.debug(f'Cleared data folder at path: {const.BASE_PATH}')

def SEPARATE():
    """
    Separate the world.
    """
    pass

def FILE(stacking_file_path):
    """
    Parse a stacking file.
    @Param stacking_file_path: Path to the stacking file.
    @Return: Parsed worlds.
    """
    return parse_stacking_file(stacking_file_path)


def ADD(
    config=None,
    image_path=None,
    amount=1,
    mode="x",
    margin=np.array([0, 0, 0]),
    worldpositionoffset=np.array([0, 0, 0]),
    worldrotationoffset=np.array([0, 0, 0]),
    worldscale=np.array([1, 1, 1]),
    amount_per_level=None,
    radie=None,
    degree=None,
):
    """
    Add floorplan to configuration.
    @Param config: Configuration file path.
    @Param image_path: Path to the image.
    @Param amount: Number of floorplans to add.
    @Param mode: Mode for stacking (x, y, z, cylinder).
    @Param margin: Margin for stacking.
    @Param worldpositionoffset: World position offset.
    @Param worldrotationoffset: World rotation offset.
    @Param worldscale: World scale.
    @Param amount_per_level: Amount per level for cylinder mode.
    @Param radie: Radius for cylinder mode.
    @Param degree: Degree for cylinder mode.
    @Return: Paths to image data.
    """
    conf = config
    if config is None:
        conf = const.IMAGE_DEFAULT_CONFIG_FILE_NAME

    if amount is None:
        amount = 1

    floorplans = []
    for _ in range(amount):
        floorplans.append(floorplan.new_floorplan(conf))

    if image_path is not None:  # Replace all image paths
        new_floorplans = []
        for f in floorplans:
            tmp_f = f
            tmp_f.image_path = image_path
            new_floorplans.append(tmp_f)
        floorplans = new_floorplans

    dir = 1
    if mode is None:
        mode = "x"
    if mode[0] == "-":
        dir = -1
        mode = mode[1]

    if LOGGING_VERBOSE:
        logger.debug(f'Adding floorplans with config: {conf}, image_path: {image_path}, amount: {amount}, mode: {mode}')

    if mode == "cylinder":
        result = execution.multiple_cylinder(
            floorplans,
            amount_per_level,
            radie,
            degree,
            world_direction=dir,
            world_position=transform.list_to_nparray(
                worldpositionoffset, np.array([0, 0, 0])
            ),
            world_rotation=transform.list_to_nparray(
                worldrotationoffset, np.array([0, 0, 0])
            ),
            world_scale=transform.list_to_nparray(worldscale),
            margin=transform.list_to_nparray(margin, np.array([0, 0, 0])),
        )
    else:
        result = execution.multiple_axis(
            floorplans,
            mode,
            dir,
            transform.list_to_nparray(margin, np.array([0, 0, 0])),
            transform.list_to_nparray(worldpositionoffset, np.array([0, 0, 0])),
            transform.list_to_nparray(worldrotationoffset, np.array([0, 0, 0])),
            transform.list_to_nparray(worldscale),
        )
    
    if LOGGING_VERBOSE:
        logger.debug(f'Added floorplans result: {result}')
    save_debug_info('ADD.txt', {
        'config': conf,
        'image_path': image_path,
        'amount': amount,
        'mode': mode,
        'margin': margin.tolist(),
        'worldpositionoffset': worldpositionoffset.tolist(),
        'worldrotationoffset': worldrotationoffset.tolist(),
        'worldscale': worldscale.tolist(),
        'amount_per_level': amount_per_level,
        'radie': radie,
        'degree': degree,
        'result': result,
    })

    return result
