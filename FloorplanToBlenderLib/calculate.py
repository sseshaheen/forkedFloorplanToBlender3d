import cv2
import math
import numpy as np
import logging
from . import detect
from . import const
from .globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE
import os

"""
Calculate
This file contains functions for handling math or calculations.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""
# Configure logging
if LOGGING_VERBOSE:
    # Load the DEBUG_SESSION_ID from the JSON file
    debug_config = load_config_from_json('./config.json')
    
    log_dir_path = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'])
    log_file_path = os.path.join(log_dir_path, 'debug.log')
    os.makedirs(os.path.dirname(log_dir_path), exist_ok=True)

    # Create a logger
    logger = logging.getLogger('debug_logger')
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

    # Create a file handler to log everything
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)



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


def average(lst):
    """
    Calculate the average of a list of numbers.
    @Param lst: List of numbers.
    @Return: Average of the numbers.
    """
    avg = sum(lst) / len(lst)
    
    if LOGGING_VERBOSE:
        logger.debug('Calculated average of list.')
    save_debug_info('average.txt', {'list': lst, 'average': avg})
    
    return avg

def points_inside_contour(points, contour):
    """
    Check if any points are inside a given contour.
    @Param points: List of points.
    @Param contour: Contour to check against.
    @Return: True if any point is inside the contour, else False.
    """
    for x, y in points:
        if cv2.pointPolygonTest(contour, (x, y), False) == 1.0:
            return True
    return False


def remove_walls_not_in_contour(walls, contour):
    """
    Remove walls that are not inside a given contour.
    @Param walls: List of wall contours.
    @Param contour: Contour to check against.
    @Return: List of walls inside the contour.
    """
    res = []
    for wall in walls:
        for point in wall:
            if points_inside_contour(point, contour):
                res.append(wall)
                break
    
    if LOGGING_VERBOSE:
        logger.debug('Removed walls not inside contour.')
    save_debug_info('remove_walls_not_in_contour.txt', {'walls': walls, 'contour': contour, 'filtered_walls': res})
    
    return res


def wall_width_average(img, image_type=None):
    """
    Calculate the average width of walls in a floorplan image.
    Used to scale the size of the image for better accuracy.
    @Param img: Input floorplan image.
    @Return: Average wall width as a float value.
    """
    if img is None:
        logging.error("ERROR: Provided image is None.")
        return None
    
    try:
        # Grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resulting image
        height, width, channels = img.shape
        blank_image = np.zeros((height, width, 3), np.uint8)  # Output image same size as original

        # Create wall image (filter out small objects from image)
        wall_img = detect.wall_filter(gray, caller='calculate_wall_width_average')
        
        # Detect walls
        boxes, img = detect.precise_boxes(wall_img, blank_image, caller='calculate_wall_width_average')

        # Filter out to only count walls
        filtered_boxes = list()
        for i, box in enumerate(boxes):
            if len(box) >= 4:  # Allow for more than 4 corners
                x, y, w, h = cv2.boundingRect(box)
                # Check for aspect ratio to identify potential walls
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Proposed aspect ratio range for walls
                    shortest = min(w, h)
                    filtered_boxes.append(shortest)
                    logging.debug(f"Box {i}: x={x}, y={y}, w={w}, h={h}, shortest={shortest}, aspect_ratio={aspect_ratio}")
                else:
                    logging.debug(f"Box {i} skipped due to aspect ratio: aspect_ratio={aspect_ratio}")
            else:
                logging.debug(f"Box {i} skipped: len(box)={len(box)}")

        # Calculate average
        if len(filtered_boxes) == 0:  # If no good boxes could be found, we use default scale
            logging.error(f"ERROR: No valid wall boxes found in the {image_type}.")
            return None

        avg_wall_width = np.mean(filtered_boxes)
        
        if LOGGING_VERBOSE:
            logger.debug(f'Calculated average wall width in {image_type}: {avg_wall_width}')
        save_debug_info(f'wall_width_average_{image_type}.txt', {'image_shape': img.shape, 'average_wall_width': avg_wall_width})
        
        return avg_wall_width

    except Exception as e:
        logging.error(f"ERROR: Exception occurred while calculating wall width average: {e}")
        return None

def best_matches_with_modulus_angle(match_list):
    """
    Compare matching matches from ORB feature matching,
    by rotating in steps over 360 degrees to find the best fit for door rotation.
    @Param match_list: List of matches from ORB feature matching.
    @Return: Indices of the best matches.
    """
    index1 = 0
    index2 = 0
    best = math.inf

    for i, _ in enumerate(match_list):
        for j, _ in enumerate(match_list):

            pos1_model = match_list[i][0]
            pos2_model = match_list[j][0]

            pos1_cap = match_list[i][1]
            pos2_cap = match_list[j][1]

            pt1 = (pos1_model[0] - pos2_model[0], pos1_model[1] - pos2_model[1])
            pt2 = (pos1_cap[0] - pos2_cap[0], pos1_cap[1] - pos2_cap[1])

            if pt1 == pt2 or pt1 == (0, 0) or pt2 == (0, 0):
                continue

            ang = math.degrees(angle_between_vectors_2d(pt1, pt2))
            diff = ang % const.DOOR_ANGLE_HIT_STEP

            if diff < best:
                best = diff
                index1 = i
                index2 = j
    
    # if LOGGING_VERBOSE:
        # logger.debug('Calculated best matches with modulus angle.')
    save_debug_info('best_matches_with_modulus_angle.txt', {'match_list': match_list, 'index1': index1, 'index2': index2})
    
    return index1, index2
def points_are_inside_or_close_to_box(door, box, caller=None):
    """
    Calculate if a point is within the vicinity of a box.
    @Param door: List of points.
    @Param box: Numpy array representing the box.
    @Return: True if any point is inside or close to the box, else False.
    """
    for point in door:
        if rect_contains_or_almost_contains_point(point, box, caller=caller):
            return True
    return False


def angle_between_vectors_2d(vector1, vector2):
    """
    Get angle between two 2D vectors.
    @Param vector1: First vector.
    @Param vector2: Second vector.
    @Return: Angle in radians.
    """
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    angle = math.acos(inner_product / (len1 * len2))
    
    # if LOGGING_VERBOSE:
        # logger.debug('Calculated angle between vectors.')
    save_debug_info('angle_between_vectors_2d.txt', {'vector1': vector1, 'vector2': vector2, 'angle': angle})
    
    return angle

def rect_contains_or_almost_contains_point(pt, box, caller=None):
    """
    Calculate if a point is within the vicinity of a box.
    @Param pt: Point to check.
    @Param box: Box to check against.
    @Return: True if the point is inside or almost inside the box, else False.
    """
    x, y, w, h = cv2.boundingRect(box)
    is_inside = x < pt[0] < x + w and y < pt[1] < y + h

    almost_inside = False

    min_dist = min(w, h)

    for point in box:
        dist = abs(point[0][0] - pt[0]) + abs(point[0][1] - pt[1])
        if dist <= min_dist:
            almost_inside = True
            break

    # if LOGGING_VERBOSE:
        # logger.debug('Checked if point is inside or almost inside box.')
    save_debug_info(f'{caller}-rect_contains_or_almost_contains_point.txt', {'point': pt, 'box': box, 'is_inside': is_inside, 'almost_inside': almost_inside})
    
    return is_inside or almost_inside

def box_center(box):
    """
    Get center position of a box.
    @Param box: Numpy array representing the box.
    @Return: Center point of the box.
    """
    x, y, w, h = cv2.boundingRect(box)
    center = (x + w / 2, y + h / 2)
    
    if LOGGING_VERBOSE:
        logger.debug('Calculated center of box.')
    save_debug_info('box_center.txt', {'box': box, 'center': center})
    
    return center

def euclidean_distance_2d(p1, p2):
    """
    Calculate Euclidean distance between two points.
    @Param p1: First point.
    @Param p2: Second point.
    @Return: Euclidean distance.
    """
    distance = math.sqrt(abs(math.pow(p1[0] - p2[0], 2) - math.pow(p1[1] - p2[1], 2)))
    
    if LOGGING_VERBOSE:
        logger.debug('Calculated Euclidean distance between points.')
    save_debug_info('euclidean_distance_2d.txt', {'point1': p1, 'point2': p2, 'distance': distance})
    
    return distance

def magnitude_2d(point):
    """
    Calculate magnitude of a 2D vector.
    @Param point: 2D vector.
    @Return: Magnitude of the vector.
    """
    magnitude = math.sqrt(point[0] * point[0] + point[1] * point[1])
    
    if LOGGING_VERBOSE:
        logger.debug('Calculated magnitude of vector.')
    save_debug_info('magnitude_2d.txt', {'point': point, 'magnitude': magnitude})
    
    return magnitude

def normalize_2d(normal):
    """
    Normalize a 2D vector.
    @Param normal: 2D vector.
    @Return: Normalized vector.
    """
    mag = magnitude_2d(normal)
    for i, val in enumerate(normal):
        normal[i] = val / mag
    
    if LOGGING_VERBOSE:
        logger.debug('Normalized 2D vector.')
    save_debug_info('normalize_2d.txt', {'vector': normal, 'magnitude': mag})
    
    return normal
