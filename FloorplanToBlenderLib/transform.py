import math
import cv2
import numpy as np
import logging
from . import const
from .globalConf import DEBUG_MODE, LOGGING_VERBOSE, DEBUG_STORAGE_PATH
import os


"""
Transform
This file contains functions for transforming data between different formats.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""

# Configure logging
logger = logging.getLogger(__name__)

def save_debug_info(filename, data):
    """
    Save debug information to a file if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        filepath  = os.path.join(DEBUG_STORAGE_PATH, filename)
        with open(filepath, 'a') as file:
            file.write(str(data))
        if LOGGING_VERBOSE:
            logger.debug(f'Saved debug info: {filepath}')

def rescale_rect(list_of_rects, scale_factor):
    """
    Rescale box relative to its center point.
    @Param list_of_rects: List of rectangles to be rescaled.
    @Param scale_factor: Factor by which to scale the rectangles.
    @Return: List of rescaled rectangles.
    """

    rescaled_rects = []
    for rect in list_of_rects:
        x, y, w, h = cv2.boundingRect(rect)

        center = (x + w / 2, y + h / 2)

        # Get center diff
        xdiff = abs(center[0] - x)
        ydiff = abs(center[1] - y)

        xshift = xdiff * scale_factor
        yshift = ydiff * scale_factor

        width = 2 * xshift
        height = 2 * yshift

        # Upper left
        new_x = x - abs(xdiff - xshift)
        new_y = y - abs(ydiff - yshift)

        # Create contour
        contour = np.array(
            [
                [[new_x, new_y]],
                [[new_x + width, new_y]],
                [[new_x + width, new_y + height]],
                [[new_x, new_y + height]],
            ]
        )
        rescaled_rects.append(contour)

    if LOGGING_VERBOSE:
        logger.debug('Rescaled rectangles.')
    save_debug_info('rescale_rect.txt', {'original_rects': list_of_rects, 'rescaled_rects': rescaled_rects})

    return rescaled_rects


def flatten(in_list):
    """
    Flatten multidimensional list into a single dimensional array.
    @Param in_list: List to be flattened.
    @Return: Flattened list.
    """
    if in_list == []:
        return []
    elif type(in_list) is not list:
        return [in_list]
    else:
        return flatten(in_list[0]) + flatten(in_list[1:])

def rotate_round_origin_vector_2d(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    @Param origin: The origin point to rotate around.
    @Param point: The point to be rotated.
    @Param angle: The angle in radians to rotate the point.
    @Return: The rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    if LOGGING_VERBOSE:
        logger.debug('Rotated point around origin.')
    save_debug_info('rotate_round_origin_vector_2d.txt', {'origin': origin, 'point': point, 'angle': angle, 'rotated_point': (qx, qy)})

    return qx, qy


def scale_model_point_to_origin(origin, point, x_scale, y_scale):
    """
    Scale 2D vector between two points.
    @Param origin: The origin point.
    @Param point: The point to be scaled.
    @Param x_scale: Scale factor in the x direction.
    @Param y_scale: Scale factor in the y direction.
    @Return: The scaled point.
    """
    dx, dy = (point[0] - origin[0], point[1] - origin[1])
    scaled_point = (dx * x_scale, dy * y_scale)

    if LOGGING_VERBOSE:
        logger.debug('Scaled point relative to origin.')
    save_debug_info('scale_model_point_to_origin.txt', {'origin': origin, 'point': point, 'x_scale': x_scale, 'y_scale': y_scale, 'scaled_point': scaled_point})

    return scaled_point

def flatten_iterative_safe(thelist, res):
    """
    Flatten list iteratively in a safe manner.
    @Param thelist: Incoming list.
    @Param res: Resulting list (initially empty).
    @Return: Flattened list.
    """
    if not thelist or not isinstance(thelist, list):
        return res
    else:
        if isinstance(thelist[0], int) or isinstance(thelist[0], float):
            res.append(thelist[0])
            return flatten_iterative_safe(thelist[1:], res)
        else:
            res.extend(flatten_iterative_safe(thelist[0], []))
            return flatten_iterative_safe(thelist[1:], res)


def verts_to_poslist(verts):
    """
    Convert verts array to a list of positions.
    @Param verts: Array of vertices.
    @Return: List of positions.
    """
    list_of_elements = flatten_iterative_safe(verts, [])  # TODO: this stopped working!

    res = []
    i = 0
    while i < len(list_of_elements) - 2:  # Might miss one vertex here!
        res.append(
            [list_of_elements[i], list_of_elements[i + 1], list_of_elements[i + 2]]
        )
        i += 3

    if LOGGING_VERBOSE:
        logger.debug('Converted verts to position list.')
    save_debug_info('verts_to_poslist.txt', {'verts': verts, 'poslist': res})

    return res


def scale_point_to_vector(boxes, pixelscale=100, height=0, scale=np.array([1, 1, 1])):
    """
    Scale a point to a vector.
    @Param boxes: List of boxes to be scaled.
    @Param pixelscale: Scale factor for pixels.
    @Param height: Height value to be added to the points.
    @Param scale: Scale vector.
    @Return: List of scaled vectors.
    """
    res = []
    for box in boxes:
        for pos in box:
            res.extend([[(pos[0]) / pixelscale, (pos[1]) / pixelscale, height]])

    if LOGGING_VERBOSE:
        logger.debug('Scaled points to vectors.')
    save_debug_info('scale_point_to_vector.txt', {'boxes': boxes, 'pixelscale': pixelscale, 'height': height, 'scale': scale, 'scaled_vectors': res})

    return res


def list_to_nparray(list, default=np.array([1, 1, 1])):
    """
    Convert a list to a numpy array.
    @Param list: List to be converted.
    @Param default: Default numpy array to use if the list is None.
    @Return: Numpy array.
    """
    if list is None:
        return default
    else:
        np_array = np.array([list[0], list[1], list[2]])

    if LOGGING_VERBOSE:
        logger.debug('Converted list to numpy array.')
    save_debug_info('list_to_nparray.txt', {'list': list, 'default': default, 'np_array': np_array})

    return np_array

def create_4xn_verts_and_faces(
    boxes,
    height=1,
    pixelscale=100,
    scale=np.array([1, 1, 1]),
    ground=False,
    ground_height=const.WALL_GROUND,
):
    """
    Create vertices and faces for horizontal objects.
    @Param boxes: List of boxes.
    @Param height: Height of the objects.
    @Param pixelscale: Scale factor for pixels.
    @Param scale: Scale vector.
    @Param ground: Boolean indicating if ground vertices should be created.
    @Param ground_height: Height of the ground.
    @Return: Verts - as [[wall1],[wall2],...] numpy array, Faces - as array to use on all boxes, Wall amount - as integer.
    """
    counter = 0
    verts = []

    # Create verts
    for box in boxes:
        verts.extend([scale_point_to_vector(box, pixelscale, height, scale)])
        if ground:
            verts.extend([scale_point_to_vector(box, pixelscale, ground_height, scale)])
        counter += 1

    faces = []

    # Create faces
    for room in verts:
        count = 0
        temp = ()
        for _ in room:
            temp = temp + (count,)
            count += 1
        faces.append([(temp)])

    if LOGGING_VERBOSE:
        logger.debug('Created 4xn vertices and faces.')
    save_debug_info('create_4xn_verts_and_faces.txt', {'boxes': boxes, 'height': height, 'pixelscale': pixelscale, 'scale': scale, 'ground': ground, 'ground_height': ground_height, 'verts': verts, 'faces': faces, 'counter': counter})

    return verts, faces, counter


def create_nx4_verts_and_faces(
    boxes, height=1, scale=np.array([1, 1, 1]), pixelscale=100, ground=const.WALL_GROUND
):
    """
    Create vertices and faces for vertical objects.
    @Param boxes: List of boxes.
    @Param height: Height of the objects.
    @Param scale: Scale vector.
    @Param pixelscale: Scale factor for pixels.
    @Param ground: Height of the ground.
    @Return: Verts - as [[wall1],[wall2],...] numpy array, Faces - as array to use on all boxes, Wall amount - as integer.
    """
    counter = 0
    verts = []

    for box in boxes:
        box_verts = []
        for index in range(0, len(box)):
            temp_verts = []
            # Get current
            current = box[index][0]

            # If last, link to first
            if len(box) - 1 >= index + 1:
                next_vert = box[index + 1][0]
            else:
                next_vert = box[0][0]  # Link to first pos

            # Create all 3D poses for each wall
            temp_verts.extend(
                [((current[0]) / pixelscale, (current[1]) / pixelscale, ground)]
            )
            temp_verts.extend(
                [
                    (
                        (current[0]) / pixelscale,
                        (current[1]) / pixelscale,
                        (height),
                    )
                ]
            )
            temp_verts.extend(
                [((next_vert[0]) / pixelscale, (next_vert[1]) / pixelscale, ground)]
            )
            temp_verts.extend(
                [
                    (
                        (next_vert[0]) / pixelscale,
                        (next_vert[1]) / pixelscale,
                        (height),
                    )
                ]
            )

            # Add wall verts to verts
            box_verts.extend([temp_verts])

            # Wall counter
            counter += 1

        verts.extend([box_verts])

    faces = [(0, 1, 3, 2)]

    if LOGGING_VERBOSE:
        logger.debug('Created nx4 vertices and faces.')
    save_debug_info('create_nx4_verts_and_faces.txt', {'boxes': boxes, 'height': height, 'scale': scale, 'pixelscale': pixelscale, 'ground': ground, 'verts': verts, 'faces': faces, 'counter': counter})

    return verts, faces, counter


def create_verts(boxes, height, pixelscale=100, scale=np.array([1, 1, 1])):
    """
    Simplified conversion of 2D positions to 3D positions, adding a height value.
    @Param boxes: 2D boxes as numpy array.
    @Param height: 3D height change.
    @Param pixelscale: Scale factor for pixels.
    @Param scale: Scale vector.
    @Return: Verts - numpy array of vectors.
    """
    verts = []

    # For each wall group
    for box in boxes:
        temp_verts = []
        # For each pos
        for pos in box:

            # Add and convert all positions
            temp_verts.extend(
                [((pos[0][0]) / pixelscale, (pos[0][1]) / pixelscale, 0.0)]
            )
            temp_verts.extend(
                [((pos[0][0]) / pixelscale, (pos[0][1]) / pixelscale, height)]
            )

        # Add box to list
        verts.extend(temp_verts)

    if LOGGING_VERBOSE:
        logger.debug('Created 3D vertices from 2D positions.')
    save_debug_info('create_verts.txt', {'boxes': boxes, 'height': height, 'pixelscale': pixelscale, 'scale': scale, 'verts': verts})

    return verts
