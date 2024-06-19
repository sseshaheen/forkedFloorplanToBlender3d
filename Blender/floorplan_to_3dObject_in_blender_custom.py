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

This code reads data from a file and creates a 3d model of that data.
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
    # Calculate center location of a mesh from verts
    x = []
    y = []
    z = []

    for vert in verts:
        x.append(vert[0])
        y.append(vert[1])
        z.append(vert[2])

    return [average(x), average(y), average(z)]


def subtract_center_verts(verts1, verts2):
    # Remove verts1 from all verts in verts2, return result, verts1 & verts2 must have same shape!
    for i in range(0, len(verts2)):
        verts2[i][0] -= verts1[0]
        verts2[i][1] -= verts1[1]
        verts2[i][2] -= verts1[2]
    return verts2


def create_custom_mesh(objname, verts, faces, mat=None, cen=None):
    """
    @Param objname, name of new mesh
    @Param pos, object position [x, y, z]
    @Param vertex, corners
    @Param faces, buildorder
    """
    # Create mesh and object
    myobject, mymesh = init_object(objname)

    # Rearrange verts to put pivot point in center of mesh
    # Find center of verts
    center = get_mesh_center(verts)
    # Subtract center from verts before creation
    proper_verts = subtract_center_verts(center, verts)

    # Generate mesh data
    mymesh.from_pydata(proper_verts, [], faces)
    # Calculate the edges
    mymesh.update(calc_edges=True)

    parent_center = [0, 0, 0]
    if cen is not None:
        parent_center = [int(cen[0] / 2), int(cen[1] / 2), int(cen[2])]

    # Move object to input verts location
    myobject.location.x = center[0] - parent_center[0]
    myobject.location.y = center[1] - parent_center[1]
    myobject.location.z = center[2] - parent_center[2]

    # add material
    if mat is None:  # add random color
        myobject.data.materials.append(
            create_mat(np.random.randint(0, 40, size=4))
        )  # add the material to the object
    else:
        myobject.data.materials.append(mat)  # add the material to the object
    return myobject


def create_mat(rgb_color):
    mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
    mat.diffuse_color = rgb_color  # change to random color
    return mat


"""
Main functionality here!
"""

def create_floorplan(base_path, program_path, name=None):

    if name is None:
        name = 0

    parent, _ = init_object("Floorplan" + str(name))

    print(f"Creating floorplan for base path: {base_path}")

    """
    Get transform data
    """

    path_to_transform_file = os.path.join(program_path, base_path, "transform")

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

    path_to_wall_vertical_faces_file = os.path.join(program_path, path_to_data, "wall_vertical_faces")
    path_to_wall_vertical_verts_file = os.path.join(program_path, path_to_data, "wall_vertical_verts")

    path_to_wall_horizontal_faces_file = os.path.join(program_path, path_to_data, "wall_horizontal_faces")
    path_to_wall_horizontal_verts_file = os.path.join(program_path, path_to_data, "wall_horizontal_verts")

    path_to_floor_faces_file = os.path.join(program_path, path_to_data, "floor_faces")
    path_to_floor_verts_file = os.path.join(program_path, path_to_data, "floor_verts")

    path_to_rooms_faces_file = os.path.join(program_path, path_to_data, "room_faces")
    path_to_rooms_verts_file = os.path.join(program_path, path_to_data, "room_verts")

    path_to_doors_vertical_faces_file = os.path.join(program_path, path_to_data, "door_vertical_faces")
    path_to_doors_vertical_verts_file = os.path.join(program_path, path_to_data, "door_vertical_verts")

    path_to_doors_horizontal_faces_file = os.path.join(program_path, path_to_data, "door_horizontal_faces")
    path_to_doors_horizontal_verts_file = os.path.join(program_path, path_to_data, "door_horizontal_verts")

    path_to_windows_vertical_faces_file = os.path.join(program_path, path_to_data, "window_vertical_faces")
    path_to_windows_vertical_verts_file = os.path.join(program_path, path_to_data, "window_vertical_verts")

    path_to_windows_horizontal_faces_file = os.path.join(program_path, path_to_data, "window_horizontal_faces")
    path_to_windows_horizontal_verts_file = os.path.join(program_path, path_to_data, "window_horizontal_verts")

    print(f"Transform data: {transform}")
    print(f"Rotation: {rot}, Position: {pos}, Scale: {scale}")

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

        print(f"Wall vertical verts: {verts}")
        print(f"Wall vertical faces: {faces}")

        # Create mesh from data
        boxcount = 0
        wallcount = 0

        # Create parent
        wall_parent, _ = init_object("Walls")

        for walls in verts:
            boxname = "Box" + str(boxcount)
            for wall in walls:
                wallname = "Wall" + str(wallcount)

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

            print(f"Created vertical walls for box: {boxname}")

        # get image top wall data
        verts = read_from_file(path_to_wall_horizontal_verts_file)
        faces = read_from_file(path_to_wall_horizontal_faces_file)

        print(f"Wall horizontal verts: {verts}")
        print(f"Wall horizontal faces: {faces}")

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

            print(f"Created horizontal walls for room: {roomname}")

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
                frame_verts = [
                    [wall[0][0], wall[0][1], wall[0][2]],
                    [wall[1][0], wall[1][1], wall[1][2]],
                    [wall[2][0], wall[2][1], wall[2][2]],
                    [wall[3][0], wall[3][1], wall[3][2]],
                    [wall[0][0], wall[0][1], wall[0][2] - 0.1],
                    [wall[1][0], wall[1][1], wall[1][2] - 0.1],
                    [wall[2][0], wall[2][1], wall[2][2] - 0.1],
                    [wall[3][0], wall[3][1], wall[3][2] - 0.1]
                ]
                frame_faces = [
                    [0, 1, 5, 4],  # Front face
                    [1, 2, 6, 5],  # Right face
                    [2, 3, 7, 6],  # Back face
                    [3, 0, 4, 7],  # Left face
                    [4, 5, 6, 7],  # Bottom face
                    [0, 1, 2, 3]   # Top face
                ]
                print(f"Frame vertices: {frame_verts}")
                print(f"Frame faces: {frame_faces}")
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
            frame_verts = [
                [verts[i][0][0], verts[i][0][1], verts[i][0][2]],
                [verts[i][1][0], verts[i][1][1], verts[i][1][2]],
                [verts[i][2][0], verts[i][2][1], verts[i][2][2]],
                [verts[i][3][0], verts[i][3][1], verts[i][3][2]],
                [verts[i][0][0], verts[i][0][1], verts[i][0][2] - 0.1],
                [verts[i][1][0], verts[i][1][1], verts[i][1][2] - 0.1],
                [verts[i][2][0], verts[i][2][1], verts[i][2][2] - 0.1],
                [verts[i][3][0], verts[i][3][1], verts[i][3][2] - 0.1]
            ]
            frame_faces = [
                [0, 1, 5, 4],  # Front face
                [1, 2, 6, 5],  # Right face
                [2, 3, 7, 6],  # Back face
                [3, 0, 4, 7],  # Left face
                [4, 5, 6, 7],  # Bottom face
                [0, 1, 2, 3]   # Top face
            ]
            print(f"Frame vertices: {frame_verts}")
            print(f"Frame faces: {frame_faces}")
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
                frame_verts = [
                    [wall[0][0], wall[0][1], wall[0][2]],
                    [wall[1][0], wall[1][1], wall[1][2]],
                    [wall[2][0], wall[2][1], wall[2][2]],
                    [wall[3][0], wall[3][1], wall[3][2]],
                    [wall[0][0], wall[0][1], wall[0][2] - 0.1],
                    [wall[1][0], wall[1][1], wall[1][2] - 0.1],
                    [wall[2][0], wall[2][1], wall[2][2] - 0.1],
                    [wall[3][0], wall[3][1], wall[3][2] - 0.1]
                ]
                frame_faces = [
                    [0, 1, 5, 4],  # Front face
                    [1, 2, 6, 5],  # Right face
                    [2, 3, 7, 6],  # Back face
                    [3, 0, 4, 7],  # Left face
                    [4, 5, 6, 7],  # Bottom face
                    [0, 1, 2, 3]   # Top face
                ]
                print(f"Frame vertices: {frame_verts}")
                print(f"Frame faces: {frame_faces}")
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
            frame_verts = [
                [verts[i][0][0], verts[i][0][1], verts[i][0][2]],
                [verts[i][1][0], verts[i][1][1], verts[i][1][2]],
                [verts[i][2][0], verts[i][2][1], verts[i][2][2]],
                [verts[i][3][0], verts[i][3][1], verts[i][3][2]],
                [verts[i][0][0], verts[i][0][1], verts[i][0][2] - 0.1],
                [verts[i][1][0], verts[i][1][1], verts[i][1][2] - 0.1],
                [verts[i][2][0], verts[i][2][1], verts[i][2][2] - 0.1],
                [verts[i][3][0], verts[i][3][1], verts[i][3][2] - 0.1]
            ]
            frame_faces = [
                [0, 1, 5, 4],  # Front face
                [1, 2, 6, 5],  # Right face
                [2, 3, 7, 6],  # Back face
                [3, 0, 4, 7],  # Left face
                [4, 5, 6, 7],  # Bottom face
                [0, 1, 2, 3]   # Top face
            ]
            print(f"Frame vertices: {frame_verts}")
            print(f"Frame faces: {frame_faces}")
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

        print(f"Floor verts: {verts}")
        print(f"Floor faces: {faces}")

        # Create mesh from data
        cornername = "Floor"
        obj = create_custom_mesh(
            cornername, verts, [faces], mat=create_mat((40, 1, 1, 1)), cen=cen
        )
        obj.parent = parent

        print(f"Created floor mesh: {cornername}")

        """
        Create rooms
        """
        # get image wall data
        verts = read_from_file(path_to_rooms_verts_file)
        faces = read_from_file(path_to_rooms_faces_file)

        print(f"Room verts: {verts}")
        print(f"Room faces: {faces}")

        # Create parent
        room_parent, _ = init_object("Rooms")

        for i in range(0, len(verts)):
            roomname = "Room" + str(i)
            obj = create_custom_mesh(roomname, verts[i], faces[i], cen=cen)
            obj.parent = room_parent

            print(f"Created room mesh: {roomname}")

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

    print("Finished creating floorplan.")

# Run the script independently with specified paths
program_path = "/home/apps/blender"  # Change this to your program path
base_path = "/home/apps/forkedFloorplanToBlender3d/Server/storage/data/7ZX5LI"  # Change this to your transform data path

# Call the function directly to create the floorplan
create_floorplan(base_path, program_path, name=0)

# Save the Blender file
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(base_path, "output_floorplan.blend"))

print("Blender file saved successfully.")