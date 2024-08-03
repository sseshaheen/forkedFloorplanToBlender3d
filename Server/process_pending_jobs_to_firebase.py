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
        result = subprocess.run(
            [blender_path, "--background", "--python", temp_script_path],
            capture_output=True, text=True, check=True
        )
        
        # Log standard output and standard error from Blender
        logging.info("Blender script executed successfully.")
        logging.info(f"Blender stdout: {result.stdout}")
        logging.info(f"Blender stderr: {result.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Blender script: {e}")
        logging.error(f"Blender stdout: {e.stdout}")
        logging.error(f"Blender stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_script_path):
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
    data_path = "/home/apps/forkedFloorplanToBlender3d/Server/storage/data"
    blender_script_template_path = "/home/apps/forkedFloorplanToBlender3d/Server/blender_script.py"
    # blender_script_template_path = "/home/apps/forkedFloorplanToBlender3d/Server/blender_script_for_testing.py"


    for job_id in os.listdir(pending_jobs_path):
        job_folder = os.path.join(pending_jobs_path, job_id)
        job_file_path = os.path.join(job_folder, "job.json")

        # Path to the Blender executable
        blender_path = "/usr/local/bin/blender"
        # blender_path = "/home/blender-4.2.0-linux-x64/blender"

        # Read the Blender script template
        with open(blender_script_template_path, "r") as file:
            script_content = file.read()

        # Replace the {job_id} placeholder with the actual job ID
        script_content = script_content.replace("{job_id}", job_id)

        print("Running Blender script...")
        run_blender_script(blender_path, script_content)

        if os.path.isdir(job_folder) and os.path.exists(job_file_path):
            logging.info(f"Processing job: {job_id}")
            with open(job_file_path, "r") as job_file:
                job_data = json.load(job_file)

            obj_file_path = os.path.join(data_path, job_id, "floorplan-adjusted.obj")
            glb_file_path = os.path.join(data_path, job_id, "floorplan-adjusted.glb")
            # this will upload the old obj file (without doors and windows):
            # obj_file_path = os.path.join(storage_path, f"{job_id}.obj")

            if os.path.exists(obj_file_path) and os.path.exists(glb_file_path):
                try:
                    # Upload the .obj file to Firebase
                    obj_url = upload_file_to_firebase(obj_file_path, job_data["obj_record"]["path"])
                    glb_url = upload_file_to_firebase(glb_file_path, job_data["obj_glb_record"]["path"])

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
                    job_data["glb_record"] = {"path": job_data["obj_record"]["path"].replace(".obj", ".glb"), "url": glb_url}
                    job_data["image_and_obj_record"]["obj_url"] = obj_url
                    job_data["image_and_obj_record"]["obj_glb_record"] = glb_url
                    # set image_successConversionTo3d to true
                    job_data["image_and_obj_record"]["image_successConversionTo3d"] = True
                    # Add the updated record
                    user_ref.update({
                        "objects": firestore.ArrayUnion([job_data["obj_record"], job_data["glb_record"]])
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_pending_jobs_to_firebase()
