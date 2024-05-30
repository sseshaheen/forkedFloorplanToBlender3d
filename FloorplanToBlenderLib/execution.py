from . import generate
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import atan2, degrees
import logging
from .globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE
import os

"""
Execution
This file contains some example usages and creations of multiple floorplans.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""

# Configure logging
if LOGGING_VERBOSE:
    # Load the DEBUG_SESSION_ID from the JSON file
    debug_config = load_config_from_json('./config.json')
    
    log_file_path = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID']) + 'debug.log'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)


    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Log warnings and above to the console

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    logger = logging.getLogger(__name__)

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
Execution
This file contains some example usages and creations of multiple floorplans.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""

def simple_single(floorplan, show=True):
    """
    Generate one simple floorplan.
    @Param floorplan: The floorplan object containing details for generating files.
    @Param show: Boolean indicating if information should be printed.
    @Return: Path to generated files.
    """
    filepath, _ = generate.generate_all_files(floorplan, show)
    
    if LOGGING_VERBOSE:
        logger.debug(f'Generated simple single floorplan: {filepath}')
    save_debug_info('simple_single.txt', {'floorplan': floorplan, 'filepath': filepath})
    
    return filepath

def multiple_axis(
    floorplans,
    axis,
    dir=1,
    margin=np.array([0, 0, 0]),
    worldpositionoffset=np.array([0, 0, 0]),
    worldrotationoffset=np.array([0, 0, 0]),
    worldscale=np.array([1, 1, 1]),
):
    """
    Generate several new apartments along axis "x","y","z".
    @Param floorplans: List of floorplan objects.
    @Param axis: Axis along which to place the floorplans ('x', 'y', 'z').
    @Param dir: Direction along the axis (1 for positive, -1 for negative).
    @Param margin: Margin to add between floorplans.
    @Param worldpositionoffset: Position offset for the world.
    @Param worldrotationoffset: Rotation offset for the world.
    @Param worldscale: Scale for the world.
    @Return: List of paths to generated data.
    """
    data_paths = []
    fshape = None

    if margin is None:
        margin = np.array([0, 0, 0])

    # for each input image path!
    for floorplan in floorplans:
        # Calculate positions and rotations here!
        if fshape is not None:
            # Generate all data for imagepath
            if axis == "y":
                filepath, fshape = generate.generate_all_files(
                    floorplan,
                    True,
                    world_direction=dir,
                    world_scale=worldscale,
                    world_position=np.array([0, fshape[1], 0]) + worldpositionoffset + margin,
                    world_rotation=worldrotationoffset,
                )
            elif axis == "x":
                filepath, fshape = generate.generate_all_files(
                    floorplan,
                    True,
                    world_scale=worldscale,
                    world_position=np.array([fshape[0], 0, 0]) + worldpositionoffset + margin,
                    world_rotation=worldrotationoffset,
                    world_direction=dir,
                )
            elif axis == "z":
                filepath, fshape = generate.generate_all_files(
                    floorplan,
                    True,
                    world_scale=worldscale,
                    world_position=np.array([0, 0, fshape[2]]) + worldpositionoffset + margin,
                    world_rotation=worldrotationoffset,
                    world_direction=dir,
                )
        else:
            filepath, fshape = generate.generate_all_files(
                floorplan,
                True,
                world_direction=dir,
                world_scale=worldscale,
                world_position=worldpositionoffset + margin,
                world_rotation=worldrotationoffset,
            )

        # add path to send to blender
        data_paths.append(filepath)

    if LOGGING_VERBOSE:
        logger.debug(f'Generated multiple floorplans along axis {axis}: {data_paths}')
    save_debug_info('multiple_axis.txt', {'floorplans': floorplans, 'axis': axis, 'dir': dir, 'margin': margin, 'worldpositionoffset': worldpositionoffset, 'worldrotationoffset': worldrotationoffset, 'worldscale': worldscale, 'data_paths': data_paths})

    return data_paths


def rotate_around_axis(axis, vec, degrees):
    """
    Rotate a vector around an axis by a given number of degrees.
    @Param axis: Axis around which to rotate.
    @Param vec: Vector to rotate.
    @Param degrees: Angle in degrees to rotate.
    @Return: Rotated vector.
    """
    rotation_radians = np.radians(degrees)
    rotation_vector = rotation_radians * axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vec)

    if LOGGING_VERBOSE:
        logger.debug('Rotated vector around axis.')
    save_debug_info('rotate_around_axis.txt', {'axis': axis, 'vec': vec, 'degrees': degrees, 'rotated_vec': rotated_vec})

    return rotated_vec

def AngleBtw2Points(pointA, pointB):
    """
    Calculate the angle between two points.
    @Param pointA: First point.
    @Param pointB: Second point.
    @Return: Angle in degrees.
    """
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    angle = degrees(atan2(changeInY, changeInX))

    if LOGGING_VERBOSE:
        logger.debug('Calculated angle between two points.')
    save_debug_info('AngleBtw2Points.txt', {'pointA': pointA, 'pointB': pointB, 'angle': angle})

    return angle

def multiple_cylinder(
    floorplans,
    amount_per_level,
    radie,
    degree,
    world_direction=None,
    world_position=np.array([0, 0, 0]),
    world_rotation=np.array([0, 0, 1]),
    world_scale=np.array([1, 1, 1]),
    margin=np.array([0, 0, 0]),
):
    """
    Generate several new apartments in a cylindrical shape.
    It is a naive solution but works for some floorplans.
    @Param floorplans: List of floorplan objects.
    @Param amount_per_level: Number of apartments per level.
    @Param radie: Radius of the cylinder.
    @Param degree: Total degrees to cover around the cylinder.
    @Param world_direction: Direction along the y-axis (default is None).
    @Param world_position: Position offset for the world.
    @Param world_rotation: Rotation vector for the world.
    @Param world_scale: Scale for the world.
    @Param margin: Margin to add between floorplans.
    @Return: List of paths to generated data.
    """
    data_paths = []
    curr_index = 0
    curr_level = 0
    degree_step = int(degree / amount_per_level)
    start_pos = (world_position[0], world_position[1] + radie, world_position[2])

    # for each input image path!
    for floorplan in floorplans:

        if curr_index == amount_per_level:
            curr_level += 1
            curr_index = 0

        curr_pos = rotate_around_axis(
            np.array([0, 0, 1]), start_pos, degree_step * curr_index
        )
        curr_pos = (int(curr_pos[0]), int(curr_pos[1]), int(curr_pos[2]))

        curr_rot = np.array([0, 0, int(degree_step * curr_index)])

        filepath, _ = generate.generate_all_files(
            floorplan,
            True,
            world_position=np.array(
                [
                    curr_pos[0] + world_position[0],
                    curr_pos[1] + world_position[1],
                    curr_level + world_position[2],
                ]
            ),
            world_rotation=curr_rot,
            world_scale=world_scale,
        )
        # add path to send to blender
        data_paths.append(filepath)

        curr_index += 1

    if LOGGING_VERBOSE:
        logger.debug('Generated multiple floorplans in cylindrical shape.')
    save_debug_info('multiple_cylinder.txt', {'floorplans': floorplans, 'amount_per_level': amount_per_level, 'radie': radie, 'degree': degree, 'world_direction': world_direction, 'world_position': world_position, 'world_rotation': world_rotation, 'world_scale': world_scale, 'margin': margin, 'data_paths': data_paths})

    return data_paths
