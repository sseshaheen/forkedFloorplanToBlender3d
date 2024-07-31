import bpy
import json
import math
import os

def read_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.loads(f.read())
    return data

def init_object(name):
    print(f"Initializing object: {name}")
    mymesh = bpy.data.meshes.new(name)
    myobject = bpy.data.objects.new(name, mymesh)
    bpy.context.collection.objects.link(myobject)
    return myobject, mymesh

def get_mesh_center(verts):
    # print(f"Calculating mesh center for verts: {verts}")
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
    # print(f"Subtracting center {verts1} from verts {verts2}")
    for i in range(0, len(verts2)):
        verts2[i][0] -= verts1[0]
        verts2[i][1] -= verts1[1]
        verts2[i][2] -= verts1[2]
    return verts2

def create_custom_mesh(objname, verts, faces, mat=None, cen=None):
    # print(f"Creating mesh for {objname} with verts: {verts} and faces: {faces}")

    # # Ensure verts is a list of lists
    # if isinstance(verts[0], float):
    #     verts = [verts]

    # Ensure faces is a list of lists of integers
    if isinstance(faces[0], list) and isinstance(faces[0][0], list):
        faces = [f[0] for f in faces]

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
    # print(f"Creating material with color: {rgb_color}")
    mat = bpy.data.materials.new(name="MaterialName")
    mat.diffuse_color = rgb_color
    return mat

def create_floorplan(base_path, program_path):
    try:
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
        },
        "windows": {
            "vertical": {
                "verts": "0window_vertical_verts.txt",
                "faces": "0window_vertical_faces.txt"
            },
            "horizontal": {
                "verts": "0window_horizontal_verts.txt",
                "faces": "0window_horizontal_faces.txt"
            }
        },
        "doors": {
            "vertical": {
                "verts": "0door_vertical_verts.txt",
                "faces": "0door_vertical_faces.txt"
            },
            "horizontal": {
                "verts": "0door_horizontal_verts.txt",
                "faces": "0door_horizontal_faces.txt"
            }
        },
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
                try:
                    verts_file = os.path.join(program_path, base_path, files["verts"])
                    faces_file = os.path.join(program_path, base_path, files["faces"])

                    if os.path.isfile(verts_file) and os.path.isfile(faces_file):
                        verts = read_from_file(verts_file)
                        faces = read_from_file(faces_file)

                        component_parent, _ = init_object(f"{component.capitalize()}{orientation.capitalize()}")

                        if isinstance(verts[0], list) and isinstance(verts[0][0], list):
                            for i, walls in enumerate(verts):
                                for j, wall in enumerate(walls):
                                    try:
                                        boxname = f"{component.capitalize()}Box{i}"
                                        wallname = f"{component.capitalize()}Wall{j}"
                                        obj = create_custom_mesh(
                                            f"{boxname}{wallname}",
                                            wall,
                                            faces,
                                            cen=cen,
                                            mat=create_mat((0.5, 0.5, 0.5, 1))
                                        )
                                        if obj:
                                            obj.parent = component_parent
                                    except Exception as e:
                                        print(f"Error creating mesh for {boxname}{wallname}: {e}")
                        else:
                            for i in range(len(verts)):
                                try:
                                    roomname = f"{component.capitalize()}{orientation.capitalize()}{i}"
                                    obj = create_custom_mesh(
                                        roomname,
                                        verts[i],
                                        [faces[i]],
                                        cen=cen,
                                        mat=create_mat((0.5, 0.5, 0.5, 1))
                                    )
                                    if obj:
                                        obj.parent = component_parent
                                except Exception as e:
                                    print(f"Error creating mesh for {roomname}: {e}")

                        component_parent.parent = parent
                except Exception as e:
                    print(f"Error processing {component} {orientation}: {e}")

        # Apply transform
        if "rotation" in transform:
            parent.rotation_euler = [math.radians(r) + (math.pi if i == 0 else 0) for i, r in enumerate(transform["rotation"])]
        if "position" in transform:
            parent.location = transform["position"]
        if "scale" in transform:
            parent.scale = transform["scale"]

    except Exception as e:
        print(f"Error in create_floorplan: {e}")
        import traceback
        traceback.print_exc()


def export_to_obj(filepath):
    bpy.ops.export_scene.obj(filepath=filepath, use_selection=False)

def main():
    try:
        program_path = bpy.path.abspath("//")
        base_path = "/home/apps/forkedFloorplanToBlender3d/Server/storage/data/{job_id}"


        # Ensure the scene is clean before adding new objects
        bpy.ops.wm.read_factory_settings(use_empty=True)

        create_floorplan(base_path, program_path)

        blend_filepath = os.path.join(base_path, "floorplan-adjusted.blend")
        obj_filepath = os.path.join(base_path, "floorplan-adjusted.obj")

        # Save the Blender file
        bpy.ops.wm.save_as_mainfile(filepath=blend_filepath)

        # Export the Blender file to OBJ format
        export_to_obj(obj_filepath)


    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
