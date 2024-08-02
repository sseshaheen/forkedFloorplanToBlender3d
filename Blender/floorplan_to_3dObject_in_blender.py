import bpy
import numpy as np
import json
import sys
import math
import os.path

"""
Floorplan to Blender

FloorplanToBlender3d
Copyright (C) 2021 Daniel Westberg

This code read data from a file and creates a 3d model of that data.
RUN THIS CODE FROM BLENDER

The new implementation starts blender and executes this script in a new project
so tutorial below can be ignored if you don't want to do this manually in blender.

HOW TO: (old style)

1. Run create script to create data files for your floorplan image.
2. Edit path in this file to generated data files.
3. Start blender
4. Open Blender text editor
5. Open this file "alt+o"
6. Run script

This code is tested on Windows 10, Blender 2.93, in December 2021.
"""

"""
Our helpful functions
"""

# TODO: restructure this file with a class and help-function to save a lot of lines of code!
# TODO: fix index should be same as floorplan folder


def read_from_file(file_path):
    """
    Read from file
    read verts data from file
    @Param file_path, path to file
    @Return data
    """
    # Now read the file back into a Python list object
    with open(file_path + ".txt", "r") as f:
        data = json.loads(f.read())
    return data


def init_object(name):
    # Create new blender object and return references to mesh and object
    mymesh = bpy.data.meshes.new(name)
    myobject = bpy.data.objects.new(name, mymesh)
    bpy.context.collection.objects.link(myobject)
    return myobject, mymesh


def average(lst):
    return sum(lst) / len(lst)

def get_mesh_center(verts):
    """
    Calculate the center of the mesh.
    @Param verts: List of vertices, expected as a list of lists.
    @Return: Center of the mesh.
    """
    if not verts:
        return [0, 0, 0]

    x, y, z = [], [], []

    for vert in verts:
        if isinstance(vert, list) and len(vert) == 3 and all(isinstance(coord, (int, float)) for coord in vert):
            x.append(vert[0])
            y.append(vert[1])
            z.append(vert[2])
        else:
            print(f"Invalid vertex format detected in get_mesh_center function.")
            print(f"Full verts list: {verts}")
            print(f"Invalid vertex: {vert}")
            raise ValueError(f"Invalid vertex format: {vert}")

    center_x = sum(x) / len(x)
    center_y = sum(y) / len(y)
    center_z = sum(z) / len(z)

    return [center_x, center_y, center_z]

def subtract_center_verts(verts1, verts2):
    # Remove verts1 from all verts in verts2, return result, verts1 & verts2 must have same shape!
    for i in range(0, len(verts2)):
        verts2[i][0] -= verts1[0]
        verts2[i][1] -= verts1[1]
        verts2[i][2] -= verts1[2]
    return verts2
def create_custom_mesh(name, verts, faces, cen=[0, 0, 0], mat=None):
    """
    Create a custom mesh in Blender from the given vertex and face data.
    @Param name: Name of the mesh.
    @Param verts: List of vertices.
    @Param faces: List of faces.
    @Param cen: Center of the mesh.
    @Param mat: Material to assign to the mesh.
    @Return: Created mesh object.
    """
    if not all(isinstance(vert, list) and len(vert) == 3 for vert in verts):
        print(f"Invalid vertex format detected before creating mesh.")
        print(f"Verts: {verts}")
        raise ValueError(f"Invalid vertex format in verts: {verts}")

    if not all(isinstance(face, list) and all(isinstance(index, int) for index in face) for face in faces):
        print(f"Invalid face format detected before creating mesh.")
        print(f"Faces: {faces}")
        raise ValueError(f"Invalid face format in faces: {faces}")

    # Create mesh data
    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Create object from mesh
    obj = bpy.data.objects.new(name=name, data=mesh)
    obj.location = cen

    # Link the object to the scene
    bpy.context.collection.objects.link(obj)

    if mat:
        obj.data.materials.append(mat)

    return obj
def create_mat(rgb_color):
    mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
    mat.diffuse_color = rgb_color  # change to random color
    return mat


"""
Main functionality here!
"""

def create_door_frame(wall, inset=0.02, thickness=0.05, depth=0.03, height_extension=0.1):
    try:
        if not isinstance(wall[0], list):
            raise TypeError(f"Expected list of vertices, got {type(wall[0])}")

        if len(wall) < 4:
            raise ValueError(f"Expected wall to have at least 4 vertices, got {len(wall)}")

        x1, y1, z1 = wall[0][0], wall[0][1], wall[0][2]
        x2, y2, z2 = wall[2][0], wall[2][1], wall[2][2]

        # Extend the height slightly
        z1 -= height_extension
        z2 += height_extension

        # Calculate direction vectors
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
        dx, dy = dx / length, dy / length
        px, py = -dy, dx

        # Create inset corners
        corners = [
            [x1 + inset * dx + inset * px, y1 + inset * dy + inset * py, z1],
            [x2 - inset * dx + inset * px, y2 - inset * dy + inset * py, z1],
            [x2 - inset * dx - inset * px, y2 - inset * dy - inset * py, z1],
            [x1 + inset * dx - inset * px, y1 + inset * dy - inset * py, z1],
        ]

        # Create frame vertices
        frame_verts = []
        for corner in corners:
            frame_verts.append(corner)
            frame_verts.append([corner[0], corner[1], z2])

        # Add threshold
        threshold_height = 0.05
        frame_verts.extend([
            [x1, y1, z1], [x2, y2, z1],
            [x1, y1, z1 + threshold_height], [x2, y2, z1 + threshold_height]
        ])

        # Create frame faces
        frame_faces = [
            [0, 1, 3, 2], [4, 6, 7, 5],  # Front and back faces
            [0, 4, 5, 1], [2, 3, 7, 6],  # Side faces
            [1, 5, 7, 3], [0, 2, 6, 4],  # Top and bottom faces
            [8, 9, 11, 10]  # Threshold
        ]

        return frame_verts, frame_faces

    except Exception as e:
        print(f"Error creating door frame: {e}")
        return [], []


def create_window_frame(wall, inset=0.02, thickness=0.05, depth=0.03, sill_depth=0.1):
    try:
        if not isinstance(wall[0], list):
            raise TypeError(f"Expected list of vertices, got {type(wall[0])}")

        if len(wall) < 4:
            raise ValueError(f"Expected wall to have at least 4 vertices, got {len(wall)}")

        x1, y1, z1 = wall[0][0], wall[0][1], wall[0][2]
        x2, y2, z2 = wall[2][0], wall[2][1], wall[2][2]

        # Calculate direction vectors
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
        dx, dy = dx / length, dy / length
        px, py = -dy, dx

        # Create inset corners
        corners = [
            [x1 + inset * dx + inset * px, y1 + inset * dy + inset * py, z1],
            [x2 - inset * dx + inset * px, y2 - inset * dy + inset * py, z1],
            [x2 - inset * dx - inset * px, y2 - inset * dy - inset * py, z1],
            [x1 + inset * dx - inset * px, y1 + inset * dy - inset * py, z1],
        ]

        # Create frame vertices
        frame_verts = []
        for corner in corners:
            frame_verts.append(corner)
            frame_verts.append([corner[0], corner[1], z2])

        # Add sill
        sill_z = z1 - 0.05  # Slightly below the bottom of the window
        frame_verts.extend([
            [x1 - sill_depth * px, y1 - sill_depth * py, sill_z],
            [x2 - sill_depth * px, y2 - sill_depth * py, sill_z],
            [x1 - sill_depth * px, y1 - sill_depth * py, z1],
            [x2 - sill_depth * px, y2 - sill_depth * py, z1]
        ])

        # Create frame faces
        frame_faces = [
            [0, 1, 5, 4],  # Front face
            [1, 2, 6, 5],  # Right face
            [2, 3, 7, 6],  # Back face
            [3, 0, 4, 7],  # Left face
            [4, 5, 6, 7],  # Bottom face
            [0, 1, 2, 3],  # Top face
            [8, 9, 11, 10], [10, 11, 3, 2]  # Sill
        ]

        return frame_verts, frame_faces

    except Exception as e:
        print(f"Error creating window frame: {e}")
        return [], []
def main(argv):

    # Remove starting object cube
    objs = bpy.data.objects
    objs.remove(objs["Cube"], do_unlink=True)

    if len(argv) > 7:  # Note YOU need 8 arguments!
        program_path = argv[5]
        target = argv[6]
    else:
        exit(0)

    """
    Instantiate
    Each argument after 7 will be a floorplan path
    """
    for i in range(7, len(argv)):
        base_path = argv[i]

        # manual debugging:
        # program_path = "/home/apps/blender"  # Change this to your program path
        # base_path = "storage/data/7ZX5LI/0"  # Change this to your transform data path

        create_floorplan(base_path, program_path, i)

    """
    Save to file
    TODO add several save modes here!
    """
    bpy.ops.wm.save_as_mainfile(filepath=program_path + target)  # "/floorplan.blend"

    """
    Send correct exit code
    """
    exit(0)


def create_floorplan(base_path, program_path, name=None):

    if name is None:
        name = 0

    parent, _ = init_object("Floorplan" + str(name))

    """
    Get transform data
    """

    path_to_transform_file = program_path + "/" + base_path + "transform"

    # read from file
    transform = read_from_file(path_to_transform_file)

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    # Calculate and move floorplan shape to center
    cen = transform["shape"]

    # Where data is stored, if shared between floorplans
    path_to_data = transform["origin_path"]

    # Set Cursor start
    bpy.context.scene.cursor.location = (0, 0, 0)

    path_to_wall_vertical_faces_file = (
        program_path + "/" + path_to_data + "wall_vertical_faces"
    )
    path_to_wall_vertical_verts_file = (
        program_path + "/" + path_to_data + "wall_vertical_verts"
    )

    path_to_wall_horizontal_faces_file = (
        program_path + "/" + path_to_data + "wall_horizontal_faces"
    )
    path_to_wall_horizontal_verts_file = (
        program_path + "/" + path_to_data + "wall_horizontal_verts"
    )

    path_to_floor_faces_file = program_path + "/" + path_to_data + "floor_faces"
    path_to_floor_verts_file = program_path + "/" + path_to_data + "floor_verts"

    path_to_rooms_faces_file = program_path + "/" + path_to_data + "room_faces"
    path_to_rooms_verts_file = program_path + "/" + path_to_data + "room_verts"

    path_to_doors_vertical_faces_file = (
        program_path + "\\" + path_to_data + "door_vertical_faces"
    )
    path_to_doors_vertical_verts_file = (
        program_path + "\\" + path_to_data + "door_vertical_verts"
    )

    path_to_doors_horizontal_faces_file = (
        program_path + "\\" + path_to_data + "door_horizontal_faces"
    )
    path_to_doors_horizontal_verts_file = (
        program_path + "\\" + path_to_data + "door_horizontal_verts"
    )

    path_to_windows_vertical_faces_file = (
        program_path + "\\" + path_to_data + "window_vertical_faces"
    )
    path_to_windows_vertical_verts_file = (
        program_path + "\\" + path_to_data + "window_vertical_verts"
    )

    path_to_windows_horizontal_faces_file = (
        program_path + "\\" + path_to_data + "window_horizontal_faces"
    )
    path_to_windows_horizontal_verts_file = (
        program_path + "\\" + path_to_data + "window_horizontal_verts"
    )

    """
    Create Walls
    """

    if (
        os.path.isfile(path_to_wall_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_wall_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_wall_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_wall_horizontal_faces_file + ".txt")
    ):
        # get image wall data
        verts = read_from_file(path_to_wall_vertical_verts_file)
        faces = read_from_file(path_to_wall_vertical_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
        wall_parent, _ = init_object("Walls")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

                if not all(isinstance(vertex, list) and len(vertex) == 3 for vertex in wall):
                    print(f"Invalid wall vertex format detected before creating custom mesh.")
                    print(f"Wall vertices: {wall}")

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.5, 0.5, 0.5, 1)),
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        # get image top wall data
        verts = read_from_file(path_to_wall_horizontal_verts_file)
        faces = read_from_file(path_to_wall_horizontal_faces_file)

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWalls" + str(i)
            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

        wall_parent.parent = parent

    """
    Create Windows
    """
    if (
        os.path.isfile(path_to_windows_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_windows_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_windows_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_windows_horizontal_faces_file + ".txt")
    ):
        print("Creating Windows...")

        # get image wall data
        verts = read_from_file(path_to_windows_vertical_verts_file)
        faces = read_from_file(path_to_windows_vertical_faces_file)

        print(f"Window vertical verts: {verts}")
        print(f"Window vertical faces: {faces}")

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
        wall_parent, _ = init_object("Windows")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            print(f"Creating box: {boxname}")
            for wall in walls:
                wallname = "Wall" + str(wallcount)
                print(f"Creating wall: {wallname}")

                # Create frame around window
                frame_verts, frame_faces = create_window_frame(wall)
                frame_obj = create_custom_mesh(
                    boxname + wallname + "Frame",
                    frame_verts,
                    frame_faces,
                    cen=cen,
                    mat=create_mat((0.3, 0.3, 0.3, 1)),
                )
                frame_obj.parent = wall_parent

                obj = create_custom_mesh(
                    boxname + wallname,
                    wall,
                    faces,
                    cen=cen,
                    mat=create_mat((0.5, 0.5, 0.5, 1)),
                )
                obj.parent = wall_parent

                wallcount += 1
            boxcount += 1

        # get windows
        verts = read_from_file(path_to_windows_horizontal_verts_file)
        faces = read_from_file(path_to_windows_horizontal_faces_file)

        print(f"Window horizontal verts: {verts}")
        print(f"Window horizontal faces: {faces}")

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        for i in range(0, len(verts)):
            roomname = "VertWindow" + str(i)
            print(f"Creating window: {roomname}")

            # Create frame around window
            frame_verts, frame_faces = create_window_frame(verts[i])
            frame_obj = create_custom_mesh(
                roomname + "Frame",
                frame_verts,
                frame_faces,
                cen=cen,
                mat=create_mat((0.3, 0.3, 0.3, 1)),
            )
            frame_obj.parent = wall_parent

            obj = create_custom_mesh(
                roomname,
                verts[i],
                faces[i],
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

        wall_parent.parent = parent
        print("Finished creating windows.")

    """
    Create Doors
    """
    if (
        os.path.isfile(path_to_doors_vertical_verts_file + ".txt")
        and os.path.isfile(path_to_doors_vertical_faces_file + ".txt")
        and os.path.isfile(path_to_doors_horizontal_verts_file + ".txt")
        and os.path.isfile(path_to_doors_horizontal_faces_file + ".txt")
    ):
        print("Creating Doors...")

        # get image wall data
        verts = read_from_file(path_to_doors_vertical_verts_file)
        faces = read_from_file(path_to_doors_vertical_faces_file)

        print(f"Door vertical verts: {verts}")
        print(f"Door vertical faces: {faces}")

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
    wall_parent, _ = init_object("Doors")

    for walls in verts:
        boxname = "Box" + str(boxcount)
        print(f"Creating box: {boxname}")
        for wall in walls:
            wallname = "Wall" + str(wallcount)
            print(f"Creating wall: {wallname}")

            # Create frame around door
            frame_verts, frame_faces = create_door_frame(wall)
            frame_obj = create_custom_mesh(
                boxname + wallname + "Frame",
                frame_verts,
                frame_faces,
                cen=cen,
                mat=create_mat((0.3, 0.3, 0.3, 1)),
            )
            frame_obj.parent = wall_parent

            obj = create_custom_mesh(
                boxname + wallname,
                wall,
                faces,
                cen=cen,
                mat=create_mat((0.5, 0.5, 0.5, 1)),
            )
            obj.parent = wall_parent

            wallcount += 1
        boxcount += 1

    # get doors
    verts = read_from_file(path_to_doors_horizontal_verts_file)
    faces = read_from_file(path_to_doors_horizontal_faces_file)

    print(f"Door horizontal verts: {verts}")
    print(f"Door horizontal faces: {faces}")

    # Create mesh from data
    boxcount = 0
    wallcount = 0

    for i in range(0, len(verts)):
        roomname = "VertDoor" + str(i)
        print(f"Creating door: {roomname}")

        # Create frame around door
        frame_verts, frame_faces = create_door_frame(verts[i])
        frame_obj = create_custom_mesh(
            roomname + "Frame",
            frame_verts,
            frame_faces,
            cen=cen,
            mat=create_mat((0.3, 0.3, 0.3, 1)),
        )
        frame_obj.parent = wall_parent

        obj = create_custom_mesh(
            roomname,
            verts[i],
            faces[i],
            cen=cen,
            mat=create_mat((0.5, 0.5, 0.5, 1)),
        )
        obj.parent = wall_parent

    wall_parent.parent = parent
    print("Finished creating doors.")

    """
    Create Floor
    """
    if os.path.isfile(path_to_floor_verts_file + ".txt") and os.path.isfile(
        path_to_floor_faces_file + ".txt"
    ):

        # get image wall data
        verts = read_from_file(path_to_floor_verts_file)
        faces = read_from_file(path_to_floor_faces_file)

        # Create mesh from data
        cornername = "Floor"
        obj = create_custom_mesh(
            cornername, verts, [faces], mat=create_mat((40, 1, 1, 1)), cen=cen
        )
        obj.parent = parent

        """
        Create rooms
        """
        # get image wall data
        verts = read_from_file(path_to_rooms_verts_file)
        faces = read_from_file(path_to_rooms_faces_file)

        # Create parent
        room_parent, _ = init_object("Rooms")

        for i in range(0, len(verts)):
            roomname = "Room" + str(i)
            obj = create_custom_mesh(roomname, verts[i], faces[i], cen=cen)
            obj.parent = room_parent

        room_parent.parent = parent

    # Perform Floorplan final position, rotation and scale
    if rot is not None:
        # compensate for mirrored image
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2]),
        ]

    if pos is not None:
        parent.location.x += pos[0]
        parent.location.y += pos[1]
        parent.location.z += pos[2]

    if scale is not None:
        parent.scale.x = scale[0]
        parent.scale.y = scale[1]
        parent.scale.z = scale[2]


if __name__ == "__main__":
    main(sys.argv)
