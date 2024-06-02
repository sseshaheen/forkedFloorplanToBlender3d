import os
import json
import shutil
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import logging

# Configure logging
log_file_path = '/home/apps/forkedFloorplanToBlender3d/logs/process_pending_jobs.log'
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

        if os.path.isdir(job_folder) and os.path.exists(job_file_path):
            logging.info(f"Processing job: {job_id}")
            with open(job_file_path, "r") as job_file:
                job_data = json.load(job_file)

            obj_file_path = os.path.join(storage_path, f"{job_id}.obj")

            if os.path.exists(obj_file_path):
                try:
                    # Upload the .obj file to Firebase
                    obj_url = upload_file_to_firebase(obj_file_path, job_data["obj_record"]["path"])
                    
                    # Update the job.json with the URL
                    job_data["obj_record"]["url"] = obj_url

                    with open(job_file_path, "w") as job_file:
                        json.dump(job_data, job_file, indent=4)

                    # Update Firestore
                    user_ref = db.collection("user_floorplans").document(job_data["userId"])
                    user_ref.update({
                        "objects": firestore.ArrayUnion([job_data["obj_record"]])
                    })

                    # Move the folder to done_jobs
                    shutil.move(job_folder, os.path.join(done_jobs_path, job_id))
                    logging.info(f"Processed job {job_id} successfully.")
                except Exception as e:
                    logging.error(f"Failed to process job {job_id}: {str(e)}")

if __name__ == "__main__":
    process_pending_jobs_to_firebase()
