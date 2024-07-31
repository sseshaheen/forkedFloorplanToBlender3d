import os
import json
import shutil
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import logging
import subprocess
import sys
import tempfile


# Configure logging
log_file_path = '/home/apps/forkedFloorplanToBlender3d/logs/process_pending_jobs_to_firebase.log'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase
cred = credentials.Certificate("/home/apps/credentials/dreamnestvr-firebase-adminsdk-j4uqg-f108fd7a39.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dreamnestvr.appspot.com'
})

# Initialize Firestore
db = firestore.client()

def run_blender_script(blender_path, script_content):
    # Create a temporary file to hold the Blender script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
        temp_script.write(script_content)
        temp_script_path = temp_script.name
    
    try:
        # Run Blender with the temporary script
        subprocess.run([blender_path, "--background", "--python", temp_script_path], check=True)
        print("Blender script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Blender script: {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary file
        os.remove(temp_script_path)


def upload_file_to_firebase(local_path: str, firebase_path: str) -> str:
    bucket = storage.bucket()
    blob = bucket.blob(firebase_path)
    blob.upload_from_filename(local_path)
    blob.make_public()  # Make the file public
    return blob.public_url

def process_pending_jobs_to_firebase():
    pending_jobs_path = "/home/apps/firebase_upload_cron/pending_jobs"
    done_jobs_path = "/home/apps/firebase_upload_cron/done_jobs"
    storage_path = "/home/apps/forkedFloorplanToBlender3d/Server/storage/objects"

    for job_id in os.listdir(pending_jobs_path):
        job_folder = os.path.join(pending_jobs_path, job_id)
        job_file_path = os.path.join(job_folder, "job.json")

        # Path to the Blender executable
        blender_path = "/usr/local/bin/blender"

            # Blender script content
        script_content = """
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
    base_path = "/home/apps/forkedFloorplanToBlender3d/Server/storage/data/"

    # create_floorplan(base_path, program_path)
    create_walls(base_path, program_path)
    create_doors(base_path, program_path)
    create_windows(base_path, program_path)
    create_others(base_path, program_path)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(base_path, job_id, "floorplan.blend"))

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

        """
        run_blender_script(blender_path, script_content)
        





        if os.path.isdir(job_folder) and os.path.exists(job_file_path):
            # logging.info(f"Processing job: {job_id}")
            with open(job_file_path, "r") as job_file:
                job_data = json.load(job_file)

            obj_file_path = os.path.join(storage_path, f"{job_id}-regenerated.obj")

            if os.path.exists(obj_file_path):
                try:
                    # Upload the .obj file to Firebase
                    obj_url = upload_file_to_firebase(obj_file_path, job_data["obj_record"]["path"])

                    with open(job_file_path, "w") as job_file:
                        json.dump(job_data, job_file, indent=4)

                    # Update Firestore
                    user_ref = db.collection("user_floorplans").document(job_data["userId"])
                    # Remove the old record if it exists
                    user_ref.update({
                        "objects": firestore.ArrayRemove([job_data["obj_record"]])
                    })
                    user_ref.update({
                        "image_and_obj": firestore.ArrayRemove([job_data["image_and_obj_record"]])
                    })
                    # Gotta do the job_data updates after the remove otherwise remove will not work
                    # Update the job.json with the URL
                    job_data["obj_record"]["url"] = obj_url
                    job_data["image_and_obj_record"]["obj_url"] = obj_url
                    # set image_successConversionTo3d to true
                    job_data["image_and_obj_record"]["image_successConversionTo3d"] = True
                    # Add the updated record
                    user_ref.update({
                        "objects": firestore.ArrayUnion([job_data["obj_record"]])
                    })
                    user_ref.update({
                        "image_and_obj": firestore.ArrayUnion([job_data["image_and_obj_record"]])
                    })

                    # Move the folder to done_jobs
                    shutil.move(job_folder, os.path.join(done_jobs_path, job_id))
                    logging.info(f"Processed job {job_id} successfully.")
                except Exception as e:
                    logging.error(f"Failed to process job {job_id}: {str(e)}")

if __name__ == "__main__":
    process_pending_jobs_to_firebase()
