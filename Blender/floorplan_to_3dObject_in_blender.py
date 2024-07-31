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
    # Calculate center location of a mesh from verts
    """
    Calculate the center of the mesh.
    @Param verts: List of vertices.
    @Return: Center of the mesh.
    """
    # Ensure that verts is a list of lists,
    # where each sublist represents a vertex with
    # three coordinates [x, y, z]
    if not verts:
        return [0, 0, 0]

    x, y, z = [], [], []

    for vert in verts:
        if isinstance(vert, list) and len(vert) == 3 and all(isinstance(coord, (int, float)) for coord in vert):
            x.append(vert[0])
            y.append(vert[1])
            z.append(vert[2])
        else:
            print(f"Invalid vertex format: {vert}")
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
    try:
        if name is None:
            name = 0

        parent, _ = init_object("Floorplan" + str(name))

        print(f"Creating floorplan: {name}")

        """
        Get transform data
        """
        try:
            path_to_transform_file = os.path.join(program_path, base_path, "transform")
            transform = read_from_file(path_to_transform_file)

            rot = transform.get("rotation")
            pos = transform.get("position")
            scale = transform.get("scale")
            cen = transform.get("shape")
            path_to_data = transform.get("origin_path", "")

            print(f"Transform data loaded: rot={rot}, pos={pos}, scale={scale}, cen={cen}")
        except Exception as e:
            print(f"Error loading transform data: {e}")
            return

        # Set Cursor start
        bpy.context.scene.cursor.location = (0, 0, 0)

        # Define file paths
        file_paths = {
            "wall_vertical": {"verts": "wall_vertical_verts", "faces": "wall_vertical_faces"},
            "wall_horizontal": {"verts": "wall_horizontal_verts", "faces": "wall_horizontal_faces"},
            "floor": {"verts": "floor_verts", "faces": "floor_faces"},
            "rooms": {"verts": "room_verts", "faces": "room_faces"},
            "doors_vertical": {"verts": "door_vertical_verts", "faces": "door_vertical_faces"},
            "doors_horizontal": {"verts": "door_horizontal_verts", "faces": "door_horizontal_faces"},
            "windows_vertical": {"verts": "window_vertical_verts", "faces": "window_vertical_faces"},
            "windows_horizontal": {"verts": "window_horizontal_verts", "faces": "window_horizontal_faces"},
        }

        for key, value in file_paths.items():
            for file_type in ["verts", "faces"]:
                file_paths[key][file_type] = os.path.join(program_path, path_to_data, value[file_type])

        """
        Create Walls
        """
        create_component(parent, file_paths["wall_vertical"], file_paths["wall_horizontal"], "Walls", cen)

        """
        Create Windows
        """
        create_component(parent, file_paths["windows_vertical"], file_paths["windows_horizontal"], "Windows", cen)

        """
        Create Doors
        """
        create_component(parent, file_paths["doors_vertical"], file_paths["doors_horizontal"], "Doors", cen)

        """
        Create Floor
        """
        create_floor(parent, file_paths["floor"], cen)

        """
        Create rooms
        """
        create_rooms(parent, file_paths["rooms"], cen)

        # Perform Floorplan final position, rotation and scale
        apply_transform(parent, rot, pos, scale)

    except Exception as e:
        print(f"Error in create_floorplan: {e}")
        import traceback
        traceback.print_exc()

def create_component(parent, vertical_files, horizontal_files, component_name, cen):
    try:
        if all(os.path.isfile(f + ".txt") for f in vertical_files.values() + horizontal_files.values()):
            print(f"Creating {component_name}...")

            component_parent, _ = init_object(component_name)

            create_meshes(vertical_files, component_parent, cen, is_vertical=True)
            create_meshes(horizontal_files, component_parent, cen, is_vertical=False)

            component_parent.parent = parent
            print(f"Finished creating {component_name}.")
    except Exception as e:
        print(f"Error creating {component_name}: {e}")

def create_meshes(files, parent, cen, is_vertical):
    try:
        verts = read_from_file(files["verts"])
        faces = read_from_file(files["faces"])

        print(f"Verts: {verts[:2]}...")  # Print first two verts
        print(f"Faces: {faces[:2]}...")  # Print first two faces

        if is_vertical:
            for i, walls in enumerate(verts):
                for j, wall in enumerate(walls):
                    create_mesh_with_frame(f"Box{i}Wall{j}", wall, faces, parent, cen)
        else:
            for i, vert in enumerate(verts):
                create_mesh_with_frame(f"Vert{parent.name}{i}", vert, faces[i], parent, cen)

    except Exception as e:
        print(f"Error creating meshes: {e}")

def create_mesh_with_frame(name, verts, faces, parent, cen):
    try:
        frame_verts = create_frame_verts(verts)
        frame_faces = [
            [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
            [3, 0, 4, 7], [4, 5, 6, 7], [0, 1, 2, 3]
        ]

        frame_obj = create_custom_mesh(
            name + "Frame", frame_verts, frame_faces, cen=cen,
            mat=create_mat((0.3, 0.3, 0.3, 1))
        )
        frame_obj.parent = parent

        obj = create_custom_mesh(
            name, verts, faces, cen=cen,
            mat=create_mat((0.5, 0.5, 0.5, 1))
        )
        obj.parent = parent
    except Exception as e:
        print(f"Error creating mesh with frame {name}: {e}")

def create_frame_verts(verts):
    return [
        verts[i] + [verts[i][0], verts[i][1], verts[i][2] - 0.1]
        for i in range(4)
    ]

def create_floor(parent, floor_files, cen):
    try:
        if all(os.path.isfile(f + ".txt") for f in floor_files.values()):
            verts = read_from_file(floor_files["verts"])
            faces = read_from_file(floor_files["faces"])

            obj = create_custom_mesh(
                "Floor", verts, [faces], mat=create_mat((40, 1, 1, 1)), cen=cen
            )
            obj.parent = parent
    except Exception as e:
        print(f"Error creating floor: {e}")

def create_rooms(parent, room_files, cen):
    try:
        if all(os.path.isfile(f + ".txt") for f in room_files.values()):
            verts = read_from_file(room_files["verts"])
            faces = read_from_file(room_files["faces"])

            room_parent, _ = init_object("Rooms")

            for i, (vert, face) in enumerate(zip(verts, faces)):
                obj = create_custom_mesh(f"Room{i}", vert, face, cen=cen)
                obj.parent = room_parent

            room_parent.parent = parent
    except Exception as e:
        print(f"Error creating rooms: {e}")

def apply_transform(parent, rot, pos, scale):
    try:
        if rot is not None:
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
    except Exception as e:
        print(f"Error applying transform: {e}")

if __name__ == "__main__":
    main(sys.argv)
