import abc
import cv2
import math
import numpy as np
import logging
from . import detect
from . import transform
from . import IO
from . import const
from . import draw
from . import calculate
from .globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE
import os

"""
Generator
This file contains structures for different floorplan detection features.

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
        # Load the DEBUG_SESSION_ID from the JSON file
        config = load_config_from_json('./config.json')

        DEBUG_STORAGE_PATH = os.path.join('./storage/debug', config['DEBUG_SESSION_ID'], '/txt')
        if not os.path.exists(DEBUG_STORAGE_PATH):
            os.makedirs(DEBUG_STORAGE_PATH)
        filepath = os.path.join(DEBUG_STORAGE_PATH, filename)
        with open(filepath, 'a') as file:
            file.write(str(data))
        if LOGGING_VERBOSE:
            logger.debug(f'Saved debug info: {filepath}')



class Generator:
    __metaclass__ = abc.ABCMeta
    # Create verts (points 3d), points to use in mesh creations
    verts = []
    # Create faces for each plane, describe order to create mesh points
    faces = []
    # Height of wall
    height = const.WALL_HEIGHT
    # Scale pixel value to 3d pos
    pixelscale = const.PIXEL_TO_3D_SCALE
    # Object scale
    scale = np.array([1, 1, 1])
    # Index is many for when there are several floorplans
    path = ""

    def __init__(self, gray, path, scale, info=False):
        self.path = path
        self.shape = self.generate(gray, info)
        self.scale = scale

    def get_shape(self, verts):
        """
        Get shape
        Rescale boxes to specified scale
        @Param verts: Input boxes
        @Return: Rescaled boxes
        """
        if len(verts) == 0:
            return [0, 0, 0]

        poslist = transform.verts_to_poslist(verts)
        high = [0, 0, 0]
        low = poslist[0]

        for pos in poslist:
            if pos[0] > high[0]:
                high[0] = pos[0]
            if pos[1] > high[1]:
                high[1] = pos[1]
            if pos[2] > high[2]:
                high[2] = pos[2]
            if pos[0] < low[0]:
                low[0] = pos[0]
            if pos[1] < low[1]:
                low[1] = pos[1]
            if pos[2] < low[2]:
                low[2] = pos[2]

        rescaled_shape = [
            (high[0] - low[0]) * self.scale[0],
            (high[1] - low[1]) * self.scale[1],
            (high[2] - low[2]) * self.scale[2],
        ]

        if LOGGING_VERBOSE:
            logger.debug('Calculated shape of verts.')
        save_debug_info('get_shape.txt', {'verts': verts, 'rescaled_shape': rescaled_shape})

        return rescaled_shape

    @abc.abstractmethod
    def generate(self, gray, info=False):
        """Perform the generation"""
        pass


class Floor(Generator):
    def __init__(self, gray, path, scale, info=False):
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        """
        Generate floor data from grayscale image.
        @Param gray: Grayscale image.
        @Param info: Boolean indicating if information should be printed.
        @Return: Shape of the floor.
        """
        # Detect outer contours (simple floor or roof solution)
        contour, _ = detect.outer_contours(gray, caller='generator_floor')

        # Create verts
        self.verts = transform.scale_point_to_vector(
            boxes=contour,
            scale=self.scale,
            pixelscale=self.pixelscale,
            height=self.height,
        )

        # Create faces
        count = 0
        for _ in self.verts:
            self.faces.extend([(count)])
            count += 1

        if info:
            print("Approximated apartment size: ", cv2.contourArea(contour))
            if LOGGING_VERBOSE:
                logger.debug(f'Approximated apartment size: {cv2.contourArea(contour)}')

        IO.save_to_file(self.path + const.FLOOR_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.FLOOR_FACES, self.faces, info)

        return self.get_shape(self.verts)


class Wall(Generator):
    def __init__(self, gray, path, scale, info=False):
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        """
        Generate wall data from grayscale image.
        @Param gray: Grayscale image.
        @Param info: Boolean indicating if information should be printed.
        @Return: Shape of the walls.
        """
        # Create wall image (filter out small objects from image)
        wall_img = detect.wall_filter(gray, caller='generator_wall')

        # Detect walls
        boxes, _ = detect.precise_boxes(wall_img, caller='generator_wall')

        # Detect contour
        contour, _ = detect.outer_contours(gray, caller='generator_wall')

        # Remove walls outside of contour
        boxes = calculate.remove_walls_not_in_contour(boxes, contour)

        # Convert boxes to verts and faces, vertically
        self.verts, self.faces, wall_amount = transform.create_nx4_verts_and_faces(
            boxes=boxes,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        if info:
            print("Walls created: ", wall_amount)
            if LOGGING_VERBOSE:
                logger.debug(f'Walls created: {wall_amount}')

        # Save data to file
        IO.save_to_file(self.path + const.WALL_VERTICAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WALL_VERTICAL_FACES, self.faces, info)

        # Convert boxes to verts and faces, horizontally
        self.verts, self.faces, wall_amount = transform.create_4xn_verts_and_faces(
            boxes=boxes,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
        )

        if info:
            print("Walls created: ", wall_amount)
            if LOGGING_VERBOSE:
                logger.debug(f'Walls created horizontally: {wall_amount}')

        # Save data to file
        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(self.path + const.WALL_HORIZONTAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WALL_HORIZONTAL_FACES, self.faces, info)

        return self.get_shape(self.verts)


class Room(Generator):
    def __init__(self, gray, path, scale, info=False):
        self.height = (
            const.WALL_HEIGHT - const.ROOM_FLOOR_DISTANCE
        )  # Place room slightly above floor
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        """
        Generate room data from grayscale image.
        @Param gray: Grayscale image.
        @Param info: Boolean indicating if information should be printed.
        @Return: Shape of the rooms.
        """
        gray = detect.wall_filter(gray, caller='generator_room')
        gray = ~gray
        rooms, colored_rooms = detect.find_rooms(gray.copy(), caller='generator_room')
        gray_rooms = cv2.cvtColor(colored_rooms, cv2.COLOR_BGR2GRAY)

        # Get box positions for rooms
        boxes, gray_rooms = detect.precise_boxes(gray_rooms, gray_rooms, caller='generator_room')

        self.verts, self.faces, counter = transform.create_4xn_verts_and_faces(
            boxes=boxes,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        if info:
            print("Number of rooms detected: ", counter)
            if LOGGING_VERBOSE:
                logger.debug(f'Number of rooms detected: {counter}')

        IO.save_to_file(self.path + const.ROOM_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.ROOM_FACES, self.faces, info)

        return self.get_shape(self.verts)


class Door(Generator):
    def __init__(self, gray, path, image_path, scale_factor, scale, info=False):
        self.image_path = image_path
        self.scale_factor = scale_factor
        super().__init__(gray, path, scale, info)

    def get_point_the_furthest_away(self, door_features, door_box):
        """
        Calculate door point furthest away from doorway.
        @Param door_features: Features of the door.
        @Param door_box: Box around the door.
        @Return: Point furthest away from the doorway.
        """
        best_point = None
        dist = 0
        center = calculate.box_center(door_box)
        for f in door_features:
            if best_point is None:
                best_point = f
                dist = abs(calculate.euclidean_distance_2d(center, f))
            else:
                distance = abs(calculate.euclidean_distance_2d(center, f))
                if dist < distance:
                    best_point = f
                    dist = distance

        if LOGGING_VERBOSE:
            logger.debug('Calculated point furthest away from doorway.')
        save_debug_info('get_point_the_furthest_away.txt', {'door_features': door_features, 'door_box': door_box, 'best_point': best_point})

        return best_point

    def get_closest_box_point_to_door_point(self, wall_point, box):
        """
        Calculate best point in box to anchor door.
        @Param wall_point: Wall point to which the door is closest.
        @Param box: Box around the door.
        @Return: Best point in box to anchor the door.
        """
        best_point = None
        dist = math.inf

        box_side_points = []
        (x, y, w, h) = cv2.boundingRect(box)

        if w < h:
            box_side_points = [[x + w / 2, y], [x + w / 2, y + h]]
        else:
            box_side_points = [[x, y + h / 2], [x + w, y + h / 2]]

        for fp in box_side_points:
            if best_point is None:
                best_point = fp
                dist = calculate.euclidean_distance_2d(wall_point, fp)
            else:
                distance = calculate.euclidean_distance_2d(wall_point, fp)
                if distance > dist:
                    best_point = fp
                    dist = distance

        if LOGGING_VERBOSE:
            logger.debug('Calculated closest box point to door point.')
        save_debug_info('get_closest_box_point_to_door_point.txt', {'wall_point': wall_point, 'box': box, 'best_point': best_point})

        return (int(best_point[0]), int(best_point[1]))

    def generate(self, gray, info=False):

        """
        Generate door data from grayscale image.
        @Param gray: Grayscale image.
        @Param info: Boolean indicating if information should be printed.
        @Return: Shape of the doors.
        """
        doors = detect.doors(self.image_path, self.scale_factor)

        door_contours = []
        # Get best door shapes!
        for door in doors:
            door_features = door[0]
            door_box = door[1]

            # Find door to space point
            space_point = self.get_point_the_furthest_away(door_features, door_box)

            # Find best box corner to use as attachment
            closest_box_point = self.get_closest_box_point_to_door_point(
                space_point, door_box
            )

            # Calculate normal
            normal_line = [
                space_point[0] - closest_box_point[0],
                space_point[1] - closest_box_point[1],
            ]

            # Normalize point
            normal_line = calculate.normalize_2d(normal_line)

            # Create door contour
            x1 = closest_box_point[0] + normal_line[1] * const.DOOR_WIDTH
            y1 = closest_box_point[1] - normal_line[0] * const.DOOR_WIDTH

            x2 = closest_box_point[0] - normal_line[1] * const.DOOR_WIDTH
            y2 = closest_box_point[1] + normal_line[0] * const.DOOR_WIDTH

            x4 = space_point[0] + normal_line[1] * const.DOOR_WIDTH
            y4 = space_point[1] - normal_line[0] * const.DOOR_WIDTH

            x3 = space_point[0] - normal_line[1] * const.DOOR_WIDTH
            y3 = space_point[1] + normal_line[0] * const.DOOR_WIDTH

            c1 = [int(x1), int(y1)]
            c2 = [int(x2), int(y2)]
            c3 = [int(x3), int(y3)]
            c4 = [int(x4), int(y4)]

            door_contour = np.array([[c1], [c2], [c3], [c4]], dtype=np.int32)
            door_contours.append(door_contour)

        if const.DEBUG_DOOR:
            print("Showing DEBUG door. Press any key to continue...")
            img = draw.contoursOnImage(gray, door_contours)
            draw.image(img)

        # Create verts for door

        self.verts, self.faces, door_amount = transform.create_nx4_verts_and_faces(
            boxes=door_contours,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        if info:
            print("Doors created: ", int(door_amount / 4))
            if LOGGING_VERBOSE:
                logger.debug(f'Doors created: {int(door_amount / 4)}')

        IO.save_to_file(self.path + "door_vertical_verts", self.verts, info)
        IO.save_to_file(self.path + "door_vertical_faces", self.faces, info)

        self.verts, self.faces, door_amount = transform.create_4xn_verts_and_faces(
            boxes=door_contours,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WALL_GROUND,
        )

        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(self.path + "door_horizontal_verts", self.verts, info)
        IO.save_to_file(self.path + "door_horizontal_faces", self.faces, info)

        return self.get_shape(self.verts)


class Window(Generator):
    # TODO: also fill small gaps between windows and walls
    # TODO: also add verts for filling gaps

    def __init__(self, gray, path, image_path, scale_factor, scale, info=False):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.scale = scale
        super().__init__(gray, path, scale, info)

    def generate(self, gray, info=False):
        """
        Generate window data from grayscale image.
        @Param gray: Grayscale image.
        @Param info: Boolean indicating if information should be printed.
        @Return: Shape of the windows.
        """
        windows = detect.windows(self.image_path, self.scale_factor)

        # Create verts for window, vertical
        v, self.faces, window_amount1 = transform.create_nx4_verts_and_faces(
            boxes=windows,
            height=const.WINDOW_MIN_MAX_GAP[0],
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=0,
        )  # Create low piece
        v2, self.faces, window_amount2 = transform.create_nx4_verts_and_faces(
            boxes=windows,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=const.WINDOW_MIN_MAX_GAP[1],
        )  # Create higher piece

        self.verts = v
        self.verts.extend(v2)
        parts_per_window = 2
        window_amount = len(v) / parts_per_window

        if info:
            print("Windows created: ", int(window_amount))
            if LOGGING_VERBOSE:
                logger.debug(f'Windows created: {int(window_amount)}')

        IO.save_to_file(self.path + const.WINDOW_VERTICAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WINDOW_VERTICAL_FACES, self.faces, info)

        # Create verts for window, horizontal

        v, f, _ = transform.create_4xn_verts_and_faces(
            boxes=windows,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WALL_GROUND,
        )
        v2, f2, _ = transform.create_4xn_verts_and_faces(
            boxes=windows,
            height=const.WINDOW_MIN_MAX_GAP[0],
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WINDOW_MIN_MAX_GAP[1],
        )

        self.verts = v
        self.verts.extend(v2)
        self.faces = f
        self.faces.extend(f2)

        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(self.path + const.WINDOW_HORIZONTAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WINDOW_HORIZONTAL_FACES, self.faces, info)

        return self.get_shape(self.verts)
