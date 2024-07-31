import bpy
import json
import math
import os

def read_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.loads(f.read())
    return data

def init_object(name):
    mymesh = bpy.data.meshes.new(name)
    myobject = bpy.data.objects.new(name, mymesh)
    bpy.context.collection.objects.link(myobject)
    return myobject, mymesh

def get_mesh_center(verts):
    if not verts:
        return [0, 0, 0]

    # Ensure verts is a list of lists
    if not isinstance(verts[0], list):
        verts = [verts]

    if not all(isinstance(v, list) and len(v) == 3 for v in verts):
        raise ValueError(f"Invalid verts format: {verts}")

    try:
        x, y, z = zip(*verts)
    except TypeError as e:
        print(f"Error unpacking verts: {verts}")
        raise e

    center_x = sum(x) / len(x)
    center_y = sum(y) / len(y)
    center_z = sum(z) / len(z)

    return [center_x, center_y, center_z]

def subtract_center_verts(verts1, verts2):
    for i in range(0, len(verts2)):
        verts2[i][0] -= verts1[0]
        verts2[i][1] -= verts1[1]
        verts2[i][2] -= verts1[2]
    return verts2

def create_custom_mesh(objname, verts, faces, mat=None, cen=None):
    print(f"Creating mesh for {objname} with verts: {verts} and faces: {faces}")

    myobject, mymesh = init_object(objname)

    center = get_mesh_center(verts)
    proper_verts = subtract_center_verts(center, verts)

    mymesh.from_pydata(proper_verts, [], faces)
    mymesh.update(calc_edges=True)

    parent_center = [0, 0, 0]
    if cen is not None:
        parent_center = [int(cen[0] / 2), int(cen[1] / 2), int(cen[2])]

    myobject.location.x = center[0] - parent_center[0]
    myobject.location.y = center[1] - parent_center[1]
    myobject.location.z = center[2] - parent_center[2]

    if mat is None:
        myobject.data.materials.append(create_mat((0.5, 0.5, 0.5, 1)))
    else:
        myobject.data.materials.append(mat)
    return myobject

def create_mat(rgb_color):
    mat = bpy.data.materials.new(name="MaterialName")
    mat.diffuse_color = rgb_color
    return mat

def main():
    program_path = bpy.path.abspath("//")
    base_path = "/home/apps/forkedFloorplanToBlender3d/Server/storage/data/{job_id}"

    # create_floorplan(base_path, program_path)
    create_walls(base_path, program_path)
    create_doors(base_path, program_path)
    create_windows(base_path, program_path)
    create_others(base_path, program_path)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(base_path, "floorplan.blend"))

def create_walls(base_path, program_path):
    parent, _ = init_object("Floorplan")

    transform_file = os.path.join(program_path, base_path, "0transform.txt")
    transform = read_from_file(transform_file)
    cen = transform["shape"]

    components = {
        "walls": {
            "vertical": {
                "verts": "0wall_vertical_verts.txt",
                "faces": "0wall_vertical_faces.txt"
            },
            "horizontal": {
                "verts": "0wall_horizontal_verts.txt",
                "faces": "0wall_horizontal_faces.txt"
            }
        }
    }

    for component, orientations in components.items():
        for orientation, files in orientations.items():
            verts_file = os.path.join(program_path, base_path, files["verts"])
            faces_file = os.path.join(program_path, base_path, files["faces"])

            if os.path.isfile(verts_file) and os.path.isfile(faces_file):
                verts = read_from_file(verts_file)
                faces = read_from_file(faces_file)

                component_parent, _ = init_object(f"{component.capitalize()}{orientation.capitalize()}")

                if isinstance(verts[0], list) and isinstance(verts[0][0], list):
                    for i, walls in enumerate(verts):
                        boxname = f"{component.capitalize()}Box{i}"
                        for j, wall in enumerate(walls):
                            wallname = f"{component.capitalize()}Wall{j}"
                            obj = create_custom_mesh(
                                f"{boxname}{wallname}",
                                wall,
                                faces,
                                cen=cen,
                                mat=create_mat((0.5, 0.5, 0.5, 1))
                            )
                            obj.parent = component_parent
                else:
                    for i in range(len(verts)):
                        roomname = f"{component.capitalize()}{orientation.capitalize()}{i}"
                        obj = create_custom_mesh(
                            roomname,
                            verts[i],
                            [faces[i]],
                            cen=cen,
                            mat=create_mat((0.5, 0.5, 0.5, 1))
                        )
                        obj.parent = component_parent

                component_parent.parent = parent

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    if rot is not None:
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2])
        ]

    if pos is not None:
        parent.location.x += pos[0]
        parent.location.y += pos[1]
        parent.location.z += pos[2]

    if scale is not None:
        parent.scale.x = scale[0]
        parent.scale.y = scale[1]
        parent.scale.z = scale[2]


def create_doors(base_path, program_path):
    parent, _ = init_object("Floorplan")

    transform_file = os.path.join(program_path, base_path, "0transform.txt")
    transform = read_from_file(transform_file)
    cen = transform["shape"]

    components = {
        "doors": {
            "vertical": {
                "verts": "0door_vertical_verts.txt",
                "faces": "0door_vertical_faces.txt"
            },
            "horizontal": {
                "verts": "0door_horizontal_verts.txt",
                "faces": "0door_horizontal_faces.txt"
            }
        }
    }

    for component, orientations in components.items():
        for orientation, files in orientations.items():
            verts_file = os.path.join(program_path, base_path, files["verts"])
            faces_file = os.path.join(program_path, base_path, files["faces"])

            if os.path.isfile(verts_file) and os.path.isfile(faces_file):
                verts = read_from_file(verts_file)
                faces = read_from_file(faces_file)

                component_parent, _ = init_object(f"{component.capitalize()}{orientation.capitalize()}")

                if isinstance(verts[0], list) and isinstance(verts[0][0], list):
                    for i, walls in enumerate(verts):
                        boxname = f"{component.capitalize()}Box{i}"
                        for j, wall in enumerate(walls):
                            wallname = f"{component.capitalize()}Wall{j}"
                            obj = create_custom_mesh(
                                f"{boxname}{wallname}",
                                wall,
                                faces,
                                cen=cen,
                                mat=create_mat((0.5, 0.5, 0.5, 1))
                            )
                            obj.parent = component_parent
                else:
                    for i in range(len(verts)):
                        roomname = f"{component.capitalize()}{orientation.capitalize()}{i}"
                        obj = create_custom_mesh(
                            roomname,
                            verts[i],
                            [faces[i]],
                            cen=cen,
                            mat=create_mat((0.5, 0.5, 0.5, 1))
                        )
                        obj.parent = component_parent

                component_parent.parent = parent

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    if rot is not None:
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2])
        ]

    if pos is not None:
        parent.location.x += pos[0]
        parent.location.y += pos[1]
        parent.location.z += pos[2]

    if scale is not None:
        parent.scale.x = scale[0]
        parent.scale.y = scale[1]
        parent.scale.z = scale[2]


def create_windows(base_path, program_path):
    parent, _ = init_object("Floorplan")

    transform_file = os.path.join(program_path, base_path, "0transform.txt")
    transform = read_from_file(transform_file)
    cen = transform["shape"]

    components = {
        "windows": {
            "vertical": {
                "verts": "0window_vertical_verts.txt",
                "faces": "0window_vertical_faces.txt"
            },
            "horizontal": {
                "verts": "0window_horizontal_verts.txt",
                "faces": "0window_horizontal_faces.txt"
            }
        }
    }

    for component, orientations in components.items():
        for orientation, files in orientations.items():
            verts_file = os.path.join(program_path, base_path, files["verts"])
            faces_file = os.path.join(program_path, base_path, files["faces"])

            if os.path.isfile(verts_file) and os.path.isfile(faces_file):
                verts = read_from_file(verts_file)
                faces = read_from_file(faces_file)

                component_parent, _ = init_object(f"{component.capitalize()}{orientation.capitalize()}")

                if isinstance(verts[0], list) and isinstance(verts[0][0], list):
                    for i, walls in enumerate(verts):
                        boxname = f"{component.capitalize()}Box{i}"
                        for j, wall in enumerate(walls):
                            wallname = f"{component.capitalize()}Wall{j}"
                            obj = create_custom_mesh(
                                f"{boxname}{wallname}",
                                wall,
                                faces,
                                cen=cen,
                                mat=create_mat((0.5, 0.5, 0.5, 1))
                            )
                            obj.parent = component_parent
                else:
                    for i in range(len(verts)):
                        roomname = f"{component.capitalize()}{orientation.capitalize()}{i}"
                        obj = create_custom_mesh(
                            roomname,
                            verts[i],
                            [faces[i]],
                            cen=cen,
                            mat=create_mat((0.5, 0.5, 0.5, 1))
                        )
                        obj.parent = component_parent

                component_parent.parent = parent

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    if rot is not None:
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2])
        ]

    if pos is not None:
        parent.location.x += pos[0]
        parent.location.y += pos[1]
        parent.location.z += pos[2]

    if scale is not None:
        parent.scale.x = scale[0]
        parent.scale.y = scale[1]
        parent.scale.z = scale[2]


def create_others(base_path, program_path):
    parent, _ = init_object("Floorplan")

    transform_file = os.path.join(program_path, base_path, "0transform.txt")
    transform = read_from_file(transform_file)
    cen = transform["shape"]

    components = {
        "floors": {
            "verts": "0floor_verts.txt",
            "faces": "0floor_faces.txt"
        },
        "rooms": {
            "verts": "0room_verts.txt",
            "faces": "0room_faces.txt"
        }
    }

    for component, orientations in components.items():
        for orientation, files in orientations.items():
            verts_file = os.path.join(program_path, base_path, files["verts"])
            faces_file = os.path.join(program_path, base_path, files["faces"])

            if os.path.isfile(verts_file) and os.path.isfile(faces_file):
                verts = read_from_file(verts_file)
                faces = read_from_file(faces_file)

                component_parent, _ = init_object(f"{component.capitalize()}{orientation.capitalize()}")

                if isinstance(verts[0], list) and isinstance(verts[0][0], list):
                    for i, walls in enumerate(verts):
                        boxname = f"{component.capitalize()}Box{i}"
                        for j, wall in enumerate(walls):
                            wallname = f"{component.capitalize()}Wall{j}"
                            obj = create_custom_mesh(
                                f"{boxname}{wallname}",
                                wall,
                                faces,
                                cen=cen,
                                mat=create_mat((0.5, 0.5, 0.5, 1))
                            )
                            obj.parent = component_parent
                else:
                    for i in range(len(verts)):
                        roomname = f"{component.capitalize()}{orientation.capitalize()}{i}"
                        obj = create_custom_mesh(
                            roomname,
                            verts[i],
                            [faces[i]],
                            cen=cen,
                            mat=create_mat((0.5, 0.5, 0.5, 1))
                        )
                        obj.parent = component_parent

                component_parent.parent = parent

    rot = transform["rotation"]
    pos = transform["position"]
    scale = transform["scale"]

    if rot is not None:
        parent.rotation_euler = [
            math.radians(rot[0]) + math.pi,
            math.radians(rot[1]),
            math.radians(rot[2])
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
    main()
