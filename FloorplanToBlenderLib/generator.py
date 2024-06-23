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


def convert_to_lists(poslist):
    """
    Convert all elements of poslist to lists, including nested structures.
    @Param poslist: Input position list
    @Return: Converted position list
    """
    def convert(element):
        if isinstance(element, (tuple, list)):
            return [convert(sub_element) for sub_element in element]
        else:
            return element

    return [convert(pos) for pos in poslist]

def flatten(poslist):
    """
    Flatten nested position lists to ensure each element is a 3D coordinate.
    @Param poslist: Input nested position list
    @Return: Flattened position list with only 3D coordinates
    """
    flat_list = []
    for pos in poslist:
        if isinstance(pos, list) and len(pos) == 3 and all(isinstance(coord, (int, float)) for coord in pos):
            flat_list.append(pos)
        elif isinstance(pos, list):
            flat_list.extend(flatten(pos))
        else:
            raise ValueError(f"Invalid position element: {pos}")
    return flat_list


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
        print(f"Original poslist: {poslist}")  # Debug print to show original poslist
        print(f"Types in original poslist: {[type(pos) for pos in poslist]}")  # Show types of elements

        # Use convert_to_lists to ensure all pos are lists
        poslist = convert_to_lists(poslist)
        poslist = flatten(poslist)
        print(f"Converted poslist: {poslist}")  # Debug print to show converted poslist
        print(f"Types in converted poslist: {[type(pos) for pos in poslist]}")  # Show types of elements

        high = [float('-inf'), float('-inf'), float('-inf')]
        low = [float('inf'), float('inf'), float('inf')]

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
            logger = configure_logging()
            if logger:
                logger.debug('Calculated shape of verts.')
        save_debug_info('get_shape.txt', {'verts': verts, 'rescaled_shape': rescaled_shape, 'poslist': poslist})

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
                logger = configure_logging()
                if logger:
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
                logger = configure_logging()
                if logger:
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
                logger = configure_logging()
                if logger:
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
                logger = configure_logging()
                if logger:
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
            logger = configure_logging()
            if logger:
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
            logger = configure_logging()
            if logger:
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
        print(f"Detected doors: {doors}")

        door_contours = []
        # Get best door shapes!
        for door in doors:
            door_features = door[0]
            door_box = door[1]
            print(f"Door features: {door_features}, Door box: {door_box}")

            # Find door to space point
            space_point = self.get_point_the_furthest_away(door_features, door_box)
            print(f"Space point: {space_point}")

            # Find best box corner to use as attachment
            closest_box_point = self.get_closest_box_point_to_door_point(space_point, door_box)
            print(f"Closest box point: {closest_box_point}")

            # Calculate normal
            normal_line = [
                space_point[0] - closest_box_point[0],
                space_point[1] - closest_box_point[1],
            ]

            # Normalize point
            normal_line = calculate.normalize_2d(normal_line)
            print(f"Normal line: {normal_line}")

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
            print(f"Door contour: {door_contour}")

        if const.DEBUG_DOOR:
            print("Showing DEBUG door. Press any key to continue...")
            img = draw.contoursOnImage(gray, door_contours)
            draw.image(img)

        # Create verts and faces for door frames
        frame_verts = []
        frame_faces = []
        for contour in door_contours:
            for i in range(4):
                frame_verts.extend([
                    [contour[i][0][0], contour[i][0][1], 0],
                    [contour[i][0][0], contour[i][0][1], 1],
                ])
                idx = len(frame_verts)
                frame_faces.append([idx - 2, idx - 1, (idx + 1) % 8, (idx + 0) % 8])
        print(f"Frame Verts: {frame_verts}")
        print(f"Frame Faces: {frame_faces}")

        self.verts, self.faces, door_amount = transform.create_nx4_verts_and_faces(
            boxes=door_contours,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
        )

        print("Vertical Door Verts: ", self.verts)
        print("Vertical Door Faces: ", self.faces)

        if info:
            print("Doors created: ", int(door_amount / 4))
            if LOGGING_VERBOSE:
                logger = configure_logging()
                if logger:
                    logger.debug(f'Doors created: {int(door_amount / 4)}')

        IO.save_to_file(self.path + "debug_door_vertical_verts", self.verts, info)
        IO.save_to_file(self.path + "debug_door_vertical_faces", self.faces, info)

        IO.save_to_file(self.path + "door_vertical_verts", self.verts + frame_verts, info)
        IO.save_to_file(self.path + "door_vertical_faces", self.faces + frame_faces, info)

        # Adding debug prints for vertical frames
        # print(f"Saved vertical door verts to {self.path + 'door_vertical_verts'}: {self.verts}")
        # print(f"Saved vertical door faces to {self.path + 'door_vertical_faces'}: {self.faces}")

        self.verts, self.faces, door_amount = transform.create_4xn_verts_and_faces(
            boxes=door_contours,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=True,
            ground_height=const.WALL_GROUND,
        )

        print("Horizontal Door Verts: ", self.verts)
        print("Horizontal Door Faces: ", self.faces)

        IO.save_to_file(self.path + "debug_door_horizontal_verts", self.verts, info)
        IO.save_to_file(self.path + "debug_door_horizontal_faces", self.faces, info)


        IO.save_to_file(self.path + "door_horizontal_verts", self.verts + frame_verts, info)
        IO.save_to_file(self.path + "door_horizontal_faces", self.faces + frame_faces, info)

        # Adding debug prints for horizontal frames
        # print(f"Saved horizontal door verts to {self.path + 'door_horizontal_verts'}: {self.verts}")
        # print(f"Saved horizontal door faces to {self.path + 'door_horizontal_faces'}: {self.faces}")

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
        print(f"Detected windows: {windows}")

        # Create verts for window, vertical
        v, faces1, window_amount1 = transform.create_nx4_verts_and_faces(
            boxes=windows,
            height=const.WINDOW_MIN_MAX_GAP[0],
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=0,
        )  # Create low piece
        v2, faces2, window_amount2 = transform.create_nx4_verts_and_faces(
            boxes=windows,
            height=self.height,
            scale=self.scale,
            pixelscale=self.pixelscale,
            ground=const.WINDOW_MIN_MAX_GAP[1],
        )  # Create higher piece

        self.verts = v + v2
        self.faces = faces1 + faces2
        parts_per_window = 2
        window_amount = len(v) / parts_per_window

        print("Vertical Window Verts: ", self.verts)
        print("Vertical Window Faces: ", self.faces)

        if info:
            print("Windows created: ", int(window_amount))
            if LOGGING_VERBOSE:
                logger = configure_logging()
                if logger:
                    logger.debug(f'Windows created: {int(window_amount)}')

        IO.save_to_file(self.path + "debug_" + const.WINDOW_VERTICAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + "debug_" + const.WINDOW_VERTICAL_FACES, self.faces, info)

        IO.save_to_file(self.path + const.WINDOW_VERTICAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WINDOW_VERTICAL_FACES, self.faces, info)

        # Create verts and faces for window frames
        frame_verts = []
        frame_faces = []
        current_index = len(self.verts)  # Starting index for new vertices

        for box in windows:
            for i in range(4):
                frame_verts.extend([
                    [box[i][0][0], box[i][0][1], 0],
                    [box[i][0][0], box[i][0][1], 1],
                ])
                idx = current_index + len(frame_verts) - 2
                frame_faces.append([idx, idx + 1, (idx + 3) % 8, (idx + 2) % 8])

        print(f"Frame Verts: {frame_verts}")
        print(f"Frame Faces: {frame_faces}")

        print("Horizontal Window Verts: ", self.verts)
        print("Horizontal Window Faces: ", self.faces)

        IO.save_to_file(self.path + "debug_" + const.WINDOW_HORIZONTAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + "debug_" + const.WINDOW_HORIZONTAL_FACES, self.faces, info)

        # Adding debug prints for vertical frames
        # print(f"Saved vertical window verts to {self.path + const.WINDOW_VERTICAL_VERTS}: {self.verts}")
        # print(f"Saved vertical window faces to {self.path + const.WINDOW_VERTICAL_FACES}: {self.faces}")

        # Adding frame vertices and faces to the original lists
        self.verts.extend(frame_verts)
        self.faces.extend(frame_faces)

        # Create verts for window, horizontal
        v, f1, _ = transform.create_4xn_verts_and_faces(
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

        self.verts.extend(v + v2)
        self.faces.extend(f1 + f2)

        IO.save_to_file(self.path + const.WINDOW_HORIZONTAL_VERTS, self.verts, info)
        IO.save_to_file(self.path + const.WINDOW_HORIZONTAL_FACES, self.faces, info)


        # Adding debug prints for horizontal frames
        # print(f"Saved horizontal window verts to {self.path + const.WINDOW_HORIZONTAL_VERTS}: {self.verts}")
        # print(f"Saved horizontal window faces to {self.path + const.WINDOW_HORIZONTAL_FACES}: {self.faces}")

        # Save final data with frames
        IO.save_to_file(self.path + "window_horizontal_verts_with_frames", self.verts, info)
        IO.save_to_file(self.path + "window_horizontal_faces_with_frames", self.faces, info)

        return self.get_shape(self.verts)
