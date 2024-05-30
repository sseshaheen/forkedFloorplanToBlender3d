import cv2
import numpy as np
from . import image
import logging
import os
from ....FloorplanToBlenderLib.globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE

"""
Testing functions before adding them to the library
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


def save_debug_image(filename, img):
    """
    Save an image to the debug directory if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        if img is None or img.size == 0:
            if LOGGING_VERBOSE:
                logger.debug(f'Cannot save debug image {filename}: image is empty or None')
            return
        
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')

        DEBUG_STORAGE_PATH = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'], 'png')
        if not os.path.exists(DEBUG_STORAGE_PATH):
            os.makedirs(DEBUG_STORAGE_PATH)

        filepath = os.path.join(DEBUG_STORAGE_PATH, filename)
        cv2.imwrite(filepath, img)
        # if LOGGING_VERBOSE:
        #     logger.debug(f'Saved debug image: {filepath}')


def find_rooms(
    img,
    noise_removal_threshold=1,
    corners_threshold=0.001,
    room_closing_max_length=10,
    gap_in_wall_threshold=500000,
    caller=None
):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """

    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal

    img[img < 128] = 0
    img[img > 128] = 255
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)

    img = ~mask

    # Detect corners (you can play with the parameters here)
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    corners = dst > corners_threshold * dst.max()

    # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
    # This gets some false positives.
    # You could try to disallow drawing through other existing lines for example.
    for y, row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):

            if x2[0] - x1[0] < room_closing_max_length:
                color = 0

                cv2.line(img, (x1[0], y), (x2[0], y), color, 1)

    for x, col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (x, y1[0]), (x, y2[0]), color, 1)

    # Mark the outside of the house as black
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    rooms = []
    for label in unique:
        component = labels == label
        if (
            img[component].sum() == 0
            or np.count_nonzero(component) < gap_in_wall_threshold
        ):
            color = 0
        else:
            rooms.append(component)
            color = np.random.randint(0, 255, size=3)
        img[component] = color

    save_debug_image(f'{caller}-find_rooms.png', img)
    return rooms, img


import os

example_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Examples/example.png"
)

# Read gray image
img = cv2.imread(example_image_path, 0)
rooms, colored_house = find_rooms(img.copy(), caller='detect_room.py')
cv2.imshow("result", colored_house)
print(rooms)
cv2.waitKey()
cv2.destroyAllWindows()
