import cv2
import numpy as np
import logging
from . import image
from . import const
from . import calculate
from . import transform
import math
import time
from .globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE
import os

# Calculate (actual) size of apartment

"""
Detect
This file contains functions used when detecting and calculating shapes in images.

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

        # # Set the root logger's level to WARNING to avoid interference
        # logging.getLogger().setLevel(logging.WARNING)

        return logger
    return None





def save_debug_image(filename, img):
    """
    Save an image to the debug directory if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        if img is None or img.size == 0:
            if LOGGING_VERBOSE:
                logger = configure_logging()
                if logger:
                    logger.debug(f'Cannot save debug image {filename}: image is empty or None')
                print(f'Cannot save debug image {filename}: image is empty or None')
            return
        
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')

        DEBUG_STORAGE_PATH = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'], 'png')
        if not os.path.exists(DEBUG_STORAGE_PATH):
            os.makedirs(DEBUG_STORAGE_PATH)
        filepath = os.path.join(DEBUG_STORAGE_PATH, filename)
        cv2.imwrite(filepath, img)
        # if LOGGING_VERBOSE:
        #     logger = configure_logging()
        #     if logger:
        #         logger.debug(f'Saved debug image: {filepath}')

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
        #     logger = configure_logging()
        #     if logger:
        #         logger.debug(f'Saved debug info: {filepath}')

def wall_filter(gray, caller=None):
    """
    Filter walls
    Filter out walls from a grayscale image.
    @Param gray: Grayscale image.
    @Return: Image of walls.
    """
    if DEBUG_MODE:
        initial_info = (
            "Starting wall_filter function\n"
            f"Input grayscale image shape: {gray.shape}\n"
            f"Caller: {caller}\n"
        )
        save_debug_info(f'{caller}-wall_filter-debug.txt', initial_info)

    _, thresh = cv2.threshold(
        gray,
        const.WALL_FILTER_TRESHOLD[0],
        const.WALL_FILTER_TRESHOLD[1],
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    save_debug_info(f'{caller}-wall_filter-debug.txt', "Applied thresholding")

    # Noise removal
    kernel = np.ones(const.WALL_FILTER_KERNEL_SIZE, np.uint8)
    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=const.WALL_FILTER_MORPHOLOGY_ITERATIONS,
    )
    save_debug_info(f'{caller}-wall_filter-debug.txt', "Performed noise removal with morphological opening")

    sure_bg = cv2.dilate(opening, kernel, iterations=const.WALL_FILTER_DILATE_ITERATIONS)
    save_debug_info(f'{caller}-wall_filter-debug.txt', "Dilated to get sure background")

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, const.WALL_FILTER_DISTANCE)
    ret, sure_fg = cv2.threshold(
        const.WALL_FILTER_DISTANCE_THRESHOLD[0] * dist_transform,
        const.WALL_FILTER_DISTANCE_THRESHOLD[1] * dist_transform.max(),
        const.WALL_FILTER_MAX_VALUE,
        const.WALL_FILTER_THRESHOLD_TECHNIQUE,
    )
    save_debug_info(f'{caller}-wall_filter-debug.txt', "Applied distance transform and thresholding for sure foreground")

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    save_debug_info(f'{caller}-wall_filter-debug.txt', "Subtracted sure foreground from sure background to get unknown regions")

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Filtered walls from the image')
    save_debug_image(f'{caller}-wall_filter.png', unknown)

    return unknown

def precise_boxes(detect_img, output_img=None, color=[100, 100, 0], caller=None):
    """
    Detect corners with boxes in image with high precision.
    @Param detect_img: Image to detect from @mandatory
    @Param output_img: Image for output (optional).
    @Param color: Color to set on output.
    @Return: corners (list of boxes), output image.
    @Source: https://stackoverflow.com/questions/50930033/drawing-lines-and-distance-to-them-on-image-opencv-python
    """
    if DEBUG_MODE:
        initial_info = (
            "Starting precise_boxes function\n"
            f"Input image shape: {detect_img.shape}\n"
            f"Output image shape: {output_img.shape if output_img is not None else 'None'}\n"
            f"Color for drawing: {color}\n"
            f"Caller: {caller}\n"
        )
        save_debug_info(f'{caller}-precise_boxes-debug.txt', initial_info)

    res = []
    contours, _ = cv2.findContours(detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    save_debug_info(f'{caller}-precise_boxes-debug.txt', f"Number of contours found: {len(contours)}")

    for i, cnt in enumerate(contours):
        epsilon = const.PRECISE_BOXES_ACCURACY * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        contour_info = (
            f"Contour {i}:\n"
            f"  epsilon = {epsilon}\n"
            f"  approx = {approx}\n"
        )
        save_debug_info(f'{caller}-precise_boxes-debug.txt', contour_info)
        if output_img is not None:
            output_img = cv2.drawContours(output_img, [approx], 0, color)
            save_debug_info(f'{caller}-precise_boxes-debug.txt', f"Contour {i} drawn on output image")

        res.append(approx)

    save_debug_info(f'{caller}-precise_boxes-debug.txt', f"Resulting corners: {res}")

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Detected precise boxes in the image')
    save_debug_image(f'{caller}-precise_boxes.png', output_img)

    return res, output_img

def __corners_and_draw_lines(img, corners_threshold, room_closing_max_length, caller=None):
    """
    Finds corners and draw lines from them.
    Help function for finding rooms.
    @Param img: Input image.
    @Param corners_threshold: Threshold for corner distance.
    @Param room_closing_max_length: Threshold for room max size.
    @Return: Output image.
    """
    # Detect corners (you can play with the parameters here)
    kernel = np.ones(const.PRECISE_HARRIS_KERNEL_SIZE, np.uint8)

    dst = cv2.cornerHarris(
        img,
        const.PRECISE_HARRIS_BLOCK_SIZE,
        const.PRECISE_HARRIS_KSIZE,
        const.PRECISE_HARRIS_K,
    )
    dst = cv2.erode(dst, kernel, iterations=const.PRECISE_ERODE_ITERATIONS)
    corners = dst > corners_threshold * dst.max()

    # Create a copy of the image for debugging purposes
    debug_img = img.copy()

    # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
    # This gets some false positives.
    # You could try to disallow drawing through other existing lines for example.
    for y, row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if x2[0] - x1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (x1[0], y), (x2[0], y), color, 1)
                # Draw in red on the debug image
                debug_color = (0, 0, 155)  # Red color in BGR format
                cv2.line(debug_img, (x1[0], y), (x2[0], y), debug_color, 1)

    for x, col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (x, y1[0]), (x, y2[0]), color, 1)
                # Draw in red on the debug image
                debug_color = (0, 0, 155)  # Red color in BGR format
                cv2.line(debug_img, (x, y1[0]), (x, y2[0]), debug_color, 1)

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Detected corners and drew lines in the image')
    save_debug_image(f'{caller}-corners_and_lines.png', debug_img)  # Save the debug image with red lines

    return img  # Return the original image


def find_rooms(
    img,
    noise_removal_threshold=const.FIND_ROOMS_NOISE_REMOVAL_THRESHOLD,
    corners_threshold=const.FIND_ROOMS_CORNERS_THRESHOLD,
    room_closing_max_length=const.FIND_ROOMS_CLOSING_MAX_LENGTH,
    gap_in_wall_min_threshold=const.FIND_ROOMS_GAP_IN_WALL_MIN_THRESHOLD,
    caller=None,
):
    """
    Detect rooms in the image.
    src: https://stackoverflow.com/questions/54274610/crop-each-of-them-using-opencv-python

    @param img: Grey scale image of rooms, already eroded and doors removed etc.
    @param noise_removal_threshold: Minimal area of blobs to be kept.
    @param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    @param room_closing_max_length: Maximum line length to add to close off open doors.
    @param gap_in_wall_min_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    @return: rooms: List of numpy arrays containing boolean masks for each detected room.
             colored_house: A colored version of the input image, where each room has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal
    
    mask = image.remove_noise(img, noise_removal_threshold, caller=f'{caller}-find_rooms')
    img = ~mask

    __corners_and_draw_lines(img, corners_threshold, room_closing_max_length, caller=f'{caller}-find_rooms')

    img, mask = image.mark_outside_black(img, mask, caller=f'{caller}-find_rooms')

    start_time = time.time()
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f"Starting find_rooms() by {caller}")

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img) #TODO: optimize since it could cause server crash or take a long time
    if LOGGING_VERBOSE:
        if logger:
            logger.debug(f"Found {ret} connected components")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    if LOGGING_VERBOSE:
        logger.debug(f"Unique labels found: {unique}")
    rooms = []

    for label in unique:
        component = labels == label
        num_nonzero = np.count_nonzero(component)

        if img_rgb[component].sum() == 0 or num_nonzero < gap_in_wall_min_threshold:
            color = 0
        else:
            rooms.append(component)
            color = np.random.randint(0, 255, size=3)

        img_rgb[component] = color

    if LOGGING_VERBOSE:
        if logger:
            logger.debug(f"find_rooms() by {caller} completed in {time.time() - start_time:.2f} seconds")
            logger.debug('Detected rooms in the image')
    save_debug_image(f'{caller}-find_rooms-rooms_detected.png', img)

    return rooms, img_rgb

def and_remove_precise_boxes(detect_img, output_img=None, color=[255, 255, 255]):
    """
    Currently not used in the main implementation
    Remove contours of detected walls from image.
    @Param detect_img: Image to detect from @mandatory
    @Param output_img: Image for output (optional).
    @Param color: Color to set on output.
    @Return: List of boxes, actual image.
    @Source: https://stackoverflow.com/questions/50930033/drawing-lines-and-distance-to-them-on-image-opencv-python
    """
    res = []
    contours, hierarchy = cv2.findContours(detect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        epsilon = const.REMOVE_PRECISE_BOXES_ACCURACY * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if output_img is not None:
            output_img = cv2.drawContours(output_img, [approx], -1, color, -1)
        res.append(approx)

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Removed precise boxes from the image')
    save_debug_image('precise_boxes_removed.png', output_img)

    return res, output_img

def outer_contours(detect_img, output_img=None, color=[255, 255, 255], caller=None):
    """
    Get the outer side of floorplan, used to get ground.
    @Param detect_img: Image to detect from  @mandatory
    @Param output_img: Image for output (optional).
    @Param color: Color to set on output.
    @Return: Approx, box.
    @Source: https://stackoverflow.com/questions/50930033/drawing-lines-and-distance-to-them-on-image-opencv-python
    """
    ret, thresh = cv2.threshold(
        detect_img,
        const.OUTER_CONTOURS_TRESHOLD[0],
        const.OUTER_CONTOURS_TRESHOLD[1],
        cv2.THRESH_BINARY_INV,
    )

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour_area = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > largest_contour_area:
            largest_contour_area = cv2.contourArea(cnt)
            largest_contour = cnt

    epsilon = const.PRECISE_BOXES_ACCURACY * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if output_img is not None:
        output_img = cv2.drawContours(output_img, [approx], 0, color)

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Detected outer contours of the floorplan')
    save_debug_image(f'{caller}-outer_contours.png', output_img)

    return approx, output_img

def doors(image_path, scale_factor):
    """
    Detect doors in the image.
    @Param image_path: Path to the image.
    @Param scale_factor: Scale factor to resize the image.
    @Return: Detected doors.
    """
    model = cv2.imread(const.DOOR_MODEL, 0)
    img = cv2.imread(image_path, 0)  # Read the image again.
    # TODO: it is not very effective to read image again here!

    img = image.cv2_rescale_image(img, scale_factor)
    _, doors = feature_match(img, model, caller='detect-doors')

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Detected doors in the image')
    save_debug_image('detect-doors-doors_detected.png', img)

    return doors

def windows(image_path, scale_factor):
    """
    Detect windows in the image.
    @Param image_path: Path to the image.
    @Param scale_factor: Scale factor to resize the image.
    @Return: Detected windows.
    """
    model = cv2.imread(const.DOOR_MODEL, 0)
    img = cv2.imread(image_path, 0)  # Read the image again.
    # TODO: it is not very effective to read image again here!
    img = image.cv2_rescale_image(img, scale_factor)
    windows, _ = feature_match(img, model, caller='detect-windows')

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Detected windows in the image')
    save_debug_image('detect-windows-windows_detected.png', img)

    return windows

def feature_match(img1, img2, caller=None):
    """
    Feature match models to floorplans in order to distinguish doors from windows.
    Also calculate where doors should exist.
    Compares result with detailed boxes and filter depending on colored pixels to deviate windows, doors, and unknowns.
    @Param img1: The first image (floorplan).
    @Param img2: The second image (model).
    @Return: Matches between the two images.
    """
    cap = img1
    model = img2
    # ORB keypoint detector
    orb = cv2.ORB_create(nfeatures=const.WINDOWS_AND_DOORS_FEATURE_N, scoreType=cv2.ORB_FAST_SCORE)
    # Create brute force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(cap, None)
    # Match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate bounds
    # these are important for group matching!
    min_x = math.inf
    min_y = math.inf
    max_x = 0
    max_y = 0

    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp_model[img1_idx].pt

        # bound checks
        if x1 < min_x:
            min_x = x1
        if x1 > max_x:
            max_x = x1

        if y1 < min_y:
            min_y = y1
        if y1 > max_y:
            max_y = y1

    # calculate min/max sizes!
    h = max_y - min_y
    w = max_x - min_x

    # Initialize lists
    list_grouped_matches = []

    # --- Create a list of objects containing matches group on nearby matches ---

    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # Get the coordinates
        # x - columns
        # y - rows
        (x1, y1) = kp_model[img1_idx].pt
        (x2, y2) = kp_frame[img2_idx].pt
        i = 0
        found = False

        for existing_match in list_grouped_matches:
            if abs(existing_match[0][1][0] - x2) < w and abs(existing_match[0][1][1] - y2) < h:
                # add to group
                list_grouped_matches[i].append(((int(x1), int(y1)), (int(x2), int(y2))))
                found = True
                break
            i += 1

        if not found:
            tmp = []
            tmp.append(((int(x1), int(y1)), (int(x2), int(y2))))
            list_grouped_matches.append(tmp)

    # Remove groups with only singles because we cant calculate rotation then!
    list_grouped_matches_filtered = [group for group in list_grouped_matches if len(group) >= const.WINDOWS_AND_DOORS_MAX_CORNERS]
    # find corners of door in model image
    corners = cv2.goodFeaturesToTrack(
        model,
        const.WINDOWS_AND_DOORS_FEATURE_TRACK_MAX_CORNERS,
        const.WINDOWS_AND_DOORS_FEATURE_TRACK_QUALITY,
        const.WINDOWS_AND_DOORS_FEATURE_TRACK_MIN_DIST
    )
    corners = np.int0(corners)

    # This is still a little hardcoded but still better than before!
    upper_left = corners[1][0]
    upper_right = corners[0][0]
    down = corners[2][0]

    max_x = max([cr[0][0] for cr in corners])
    max_y = max([cr[0][1] for cr in corners])
    min_x = min([cr[0][0] for cr in corners])
    min_y = min([cr[0][1] for cr in corners])

    origin = (int((max_x + min_x) / 2), int((min_y + max_y) / 2))

    list_of_proper_transformed_doors = []

    # Calculate position and rotation of doors
    for match in list_grouped_matches_filtered:

        # calculate offsets from points
        index1, index2 = calculate.best_matches_with_modulus_angle(match)

        pos1_model = match[index1][0]
        pos2_model = match[index2][0]

        # calculate actual position from offsets with rotation!
        pos1_cap = match[index1][1]
        pos2_cap = match[index2][1]

        pt1 = (pos1_model[0] - pos2_model[0], pos1_model[1] - pos2_model[1])
        pt2 = (pos1_cap[0] - pos2_cap[0], pos1_cap[1] - pos2_cap[1])

        ang = math.degrees(calculate.angle_between_vectors_2d(pt1, pt2))

        # rotate door
        new_upper_left = transform.rotate_round_origin_vector_2d(origin, upper_left, math.radians(ang))
        new_upper_right = transform.rotate_round_origin_vector_2d(origin, upper_right, math.radians(ang))
        new_down = transform.rotate_round_origin_vector_2d(origin, down, math.radians(ang))
        new_pos1_model = transform.rotate_round_origin_vector_2d(origin, pos1_model, math.radians(ang))

        # calculate scale, and rescale model
        """
        # TODO: fix this scaling problem!
        new_cap1 = rotate(origin, pos1_cap, math.radians(ang))
        new_cap2 = rotate(origin, pos2_cap, math.radians(ang))
        new_model1 = rotate(origin, pos1_model, math.radians(ang))
        new_model2 = rotate(origin, pos2_model, math.radians(ang))

        cap_size = [(new_cap1[0]- new_cap2[0]), (new_cap1[1]- new_cap2[1])]
        model_size = [(new_model1[0]-new_model2[0]),(new_model1[1]-new_model2[1])]
        
        
        if cap_size[1] != 0 or model_size[1] != 0:
            x_scale = abs(cap_size[0]/model_size[0])
            y_scale = abs(cap_size[1]/model_size[1])
            print(x_scale, y_scale)
            scaled_upper_left = scale_model_point_to_origin( origin, new_upper_left,x_scale, y_scale)
            #scaled_upper_right = scale_model_point_to_origin( origin, new_upper_right,x_scale, y_scale)
            #scaled_down = scale_model_point_to_origin( origin, new_down,x_scale, y_scale)
            scaled_pos1_model = scale_model_point_to_origin( origin, new_pos1_model,x_scale, y_scale)
        else:
        """
        scaled_upper_left = new_upper_left
        scaled_upper_right = new_upper_right
        scaled_down = new_down
        scaled_pos1_model = new_pos1_model

        offset = (scaled_pos1_model[0] - pos1_model[0], scaled_pos1_model[1] - pos1_model[1])
        # calculate dist!
        move_dist = (pos1_cap[0] - pos1_model[0], pos1_cap[1] - pos1_model[1])

        # draw corners!
        moved_new_upper_left = (int(scaled_upper_left[0] + move_dist[0] - offset[0]), int(scaled_upper_left[1] + move_dist[1] - offset[1]))
        moved_new_upper_right = (int(scaled_upper_right[0] + move_dist[0] - offset[0]), int(scaled_upper_right[1] + move_dist[1] - offset[1]))
        moved_new_down = (int(scaled_down[0] + move_dist[0] - offset[0]), int(scaled_down[1] + move_dist[1] - offset[1]))

        list_of_proper_transformed_doors.append([moved_new_upper_left, moved_new_upper_right, moved_new_down])

    gray = wall_filter(img1, caller=f'{caller}-feature_match')
    gray = ~gray  # TODO: is it necessary to convert to grayscale again?
    rooms, colored_rooms = find_rooms(gray.copy(), caller=f'{caller}-feature_match')
    doors, colored_doors = find_details(gray.copy(), caller=f'{caller}-feature_match')
    gray_rooms = cv2.cvtColor(colored_doors, cv2.COLOR_BGR2GRAY)

    # get box positions for rooms
    boxes, gray_rooms = precise_boxes(gray_rooms, caller=f'{caller}-feature_match')

    windows = []
    doors = []
    # classify boxes
    # window, door, none
    for box in boxes:

        # is a door inside box?
        is_door = False
        _door = []
        for door in list_of_proper_transformed_doors:
            if calculate.points_are_inside_or_close_to_box(door, box, caller=f'{caller}-feature_match'):
            # TODO: match door with only one box, the closest one!
                is_door = True
                _door = door
                break

        if is_door:
            doors.append((_door, box))
            continue

        # is window?
        x, y, w, h = cv2.boundingRect(box)
        cropped = img1[y: y + h, x: x + w]
        # bandpassfilter
        total = np.sum(cropped)
        colored = np.sum(cropped > 0)
        low = const.WINDOWS_COLORED_PIXELS_THRESHOLD[0]
        high = const.WINDOWS_COLORED_PIXELS_THRESHOLD[1]

        amount_of_colored = colored / total

        if low < amount_of_colored < high:
            windows.append(box)

    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug('Feature matched the models to the floorplan')
    save_debug_image(f'{caller}-feature_matched.png', img1)

    return transform.rescale_rect(windows, const.WINDOWS_RESCALE_TO_FIT), doors

def find_details(
    img,
    noise_removal_threshold=const.DETAILS_NOISE_REMOVAL_THRESHOLD,
    corners_threshold=const.DETAILS_CORNERS_THRESHOLD,
    room_closing_max_length=const.DETAILS_CLOSING_MAX_LENGTH,
    gap_in_wall_max_threshold=const.DETAILS_GAP_IN_WALL_THRESHOLD[1],
    gap_in_wall_min_threshold=const.DETAILS_GAP_IN_WALL_THRESHOLD[0],
    caller=None
):
    """
    Detect details in the image.
    I have copied and changed this function some...
    origin from
    https://stackoverflow.com/questions/54274610/crop-each-of-them-using-opencv-python
    @Param img: Grey scale image of rooms, already eroded and doors removed etc.
    @Param noise_removal_threshold: Minimal area of blobs to be kept.
    @Param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    @Param room_closing_max_length: Maximum line length to add to close off open doors.
    @Param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    @Return: details: List of numpy arrays containing boolean masks for each detected detail.
             colored_house: A colored version of the input image, where each detail has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal

    mask = image.remove_noise(img, noise_removal_threshold, caller=f'{caller}-find_details')
    img = ~mask

    __corners_and_draw_lines(img, corners_threshold, room_closing_max_length, caller=f'{caller}-find_details')

    img, mask = image.mark_outside_black(img, mask, caller=f'{caller}-find_details')

    # Find the connected components in the house
    start_time = time.time()
    if LOGGING_VERBOSE:
        logger = configure_logging()
        if logger:
            logger.debug(f"Starting find_details() by {caller}")
    ret, labels = cv2.connectedComponents(img) #TODO: optimize since it could cause server crash or take a long time
    logger.debug(f"Found {ret} connected components")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    if LOGGING_VERBOSE:
        logger.debug(f"Unique labels found: {unique}")

    details = []
    for label in unique:
        component = labels == label
        num_nonzero = np.count_nonzero(component)

        if (
            img_rgb[component].sum() == 0
            or num_nonzero < gap_in_wall_min_threshold
            or num_nonzero > gap_in_wall_max_threshold
        ):
            color = 0
        else:
            details.append(component)
            color = np.random.randint(0, 255, size=3)

        img_rgb[component] = color

    if LOGGING_VERBOSE:
        if logger:
            logger.debug(f"find_details() by {caller} completed in {time.time() - start_time:.2f} seconds")
            logger.debug('Detected details in the image')
    save_debug_image(f'{caller}-details_detected.png', img)

    return details, img
