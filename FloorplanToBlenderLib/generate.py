from . import IO
from . import const
from . import transform
import numpy as np
import logging
from FloorplanToBlenderLib.generator import Door, Floor, Room, Wall, Window
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
Generate
This file contains code for generate data files, used when creating blender project.
A temp storage of calculated data and a way to transfer data to the blender script.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""


def generate_all_files(
    floorplan,
    info,
    world_direction=None,
    world_scale=np.array([1, 1, 1]),
    world_position=np.array([0, 0, 0]),
    world_rotation=np.array([0, 0, 0]),
):
    """
    Generate all data files
    @Param floorplan: The floorplan object containing details for generating files.
    @Param info: Boolean indicating if information should be printed.
    @Param world_direction: Direction for building (default is None, set to 1).
    @Param world_scale: Scale vector for the world.
    @Param world_position: Position vector  of float
    @Param world_rotation: Rotation vector  of float
    @Return: Path to the generated file, shape of the generated object.
    """
    if world_direction is None:
        world_direction = 1

    scale = [
        floorplan.scale[0] * world_scale[0],
        floorplan.scale[1] * world_scale[1],
        floorplan.scale[2] * world_scale[2],
    ]

    if info:
        logger.info(
            " ----- Generate ",
            floorplan.image_path,
            " at pos ",
            transform.list_to_nparray(floorplan.position) + transform.list_to_nparray(world_position),
            " rot ",
            transform.list_to_nparray(floorplan.rotation) + transform.list_to_nparray(world_rotation),
            " scale ",
            scale,
            " -----",
        )

    # Get path to save data
    path = IO.create_new_floorplan_path(const.BASE_PATH)

    origin_path, shape = IO.find_reuseable_data(floorplan.image_path, const.BASE_PATH)

    if origin_path is None:
        origin_path = path

        _, gray, scale_factor = IO.read_image(floorplan.image_path, floorplan)

        if floorplan.floors:
            shape = Floor(gray, path, scale, info).shape

        if floorplan.walls:
            if shape is not None:
                new_shape = Wall(gray, path, scale, info).shape
                shape = validate_shape(shape, new_shape)
            else:
                shape = Wall(gray, path, scale, info).shape

        if floorplan.rooms:
            if shape is not None:
                new_shape = Room(gray, path, scale, info).shape
                shape = validate_shape(shape, new_shape)
            else:
                shape = Room(gray, path, scale, info).shape

        if floorplan.windows:
            Window(gray, path, floorplan.image_path, scale_factor, scale, info)

        if floorplan.doors:
            Door(gray, path, floorplan.image_path, scale_factor, scale, info)

    generate_transform_file(
        floorplan.image_path,
        path,
        info,
        floorplan.position,
        world_position,
        floorplan.rotation,
        world_rotation,
        scale,
        shape,
        path,
        origin_path,
    )

    if floorplan.position is not None:
        shape = [
            world_direction * shape[0] + floorplan.position[0] + world_position[0],
            world_direction * shape[1] + floorplan.position[1] + world_position[1],
            world_direction * shape[2] + floorplan.position[2] + world_position[2],
        ]

    if shape is None:
        shape = [0, 0, 0]

    if LOGGING_VERBOSE:
        logger.debug(f'Generated all files for floorplan: {floorplan.image_path}')
    save_debug_info('generate_all_files.txt', {'path': path, 'shape': shape})

    return path, shape


def validate_shape(old_shape, new_shape):
    """
    Validate shape, use this to calculate an object's total shape.
    @Param old_shape: The old shape of the object.
    @Param new_shape: The new shape of the object.
    @Return: Total shape combining old and new shapes.
    """
    shape = [0, 0, 0]
    shape[0] = max(old_shape[0], new_shape[0])
    shape[1] = max(old_shape[1], new_shape[1])
    shape[2] = max(old_shape[2], new_shape[2])

    if LOGGING_VERBOSE:
        logger.debug(f'Validated shape: old_shape={old_shape}, new_shape={new_shape}, combined_shape={shape}')
    save_debug_info('validate_shape.txt', {'old_shape': old_shape, 'new_shape': new_shape, 'combined_shape': shape})

    return shape


def generate_transform_file(
    img_path,
    path,
    info,
    position,
    world_position,
    rotation,
    world_rotation,
    scale,
    shape,
    data_path,
    origin_path,
):
    """
    Generate transform file.
    A transform contains information about an object's position, rotation.
    @Param img_path: Path to the image.
    @Param path: Path to save the transform file.
    @Param info: Boolean indicating if information should be printed.
    @Param position: Position vector.
    @Param world_position: World position vector.
    @Param rotation: Rotation vector.
    @Param world_rotation: World rotation vector.
    @Param scale: Scale vector.
    @Param shape: Shape of the object.
    @Param data_path: Path to the data.
    @Param origin_path: Path to the origin data.
    @Return: Transform dictionary.
    """
    # Create transform map
    transform = {}
    if position is None:
        transform[const.STR_POSITION] = np.array([0, 0, 0])
    else:
        transform[const.STR_POSITION] = position + world_position

    if scale is None:
        transform["scale"] = np.array([1, 1, 1])
    else:
        transform["scale"] = scale

    if rotation is None:
        transform[const.STR_ROTATION] = np.array([0, 0, 0])
    else:
        transform[const.STR_ROTATION] = rotation + world_rotation

    if shape is None:
        transform[const.STR_SHAPE] = np.array([0, 0, 0])
    else:
        transform[const.STR_SHAPE] = shape

    transform[const.STR_IMAGE_PATH] = img_path

    transform[const.STR_ORIGIN_PATH] = origin_path

    transform[const.STR_DATA_PATH] = data_path

    IO.save_to_file(path + "transform", transform, info)

    if LOGGING_VERBOSE:
        logger.debug(f'Generated transform file for image: {img_path}')
    save_debug_info('generate_transform_file.txt', transform)

    return transform
