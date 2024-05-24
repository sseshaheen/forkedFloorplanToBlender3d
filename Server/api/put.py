from typing import Tuple, Dict
from api.api import Api
from api.post import Post  # needed to call transform function
import firebase_admin
from firebase_admin import credentials, storage, firestore
import os
import uuid
from datetime import datetime
import time
import requests


# Path to the downloaded service account key
# TODO: verify incoming userId through appropriate authentcation received from the mobile app
# change the file from public to private in "upload_file_to_firebase"
# TODO: delete old files after certain interval so as not to fill up the server disk
cred = credentials.Certificate("/home/apps/credentials/dreamnestvr-firebase-adminsdk-j4uqg-f108fd7a39.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'dreamnestvr.appspot.com'
})


"""
FloorplanToBlender3d
Copyright (C) 2021 Daniel Westberg
"""


def create_file(ref, id, iformat, file):
    """Write incoming data to file"""
    file_path = ref.shared.parentPath + "/" + ref.shared.imagesPath + "/" + id + iformat
    open(file_path, "wb").write(file)


class Put(Api):
    def __init__(self, client, shared_variables):
        super().__init__(client, shared_variables)
        # All all viable functions here!
        self.dispatched_calls["create"] = self.create
        self.dispatched_calls["createandtransform"] = self.createandtransform

    def create(
        self, id: str, hash: str, iformat: str, file: bytes, *args, **kwargs
    ) -> Tuple[str, bool]:
        """
        Upload new image to server.
        @Return List[ response, status]
        """
        # id and hash correct exist?
        status = True
        if (id, hash, False) in self.shared.all_ids:

            # format supported?
            if (
                iformat in self.shared.supported_image_formats
                or iformat in self.shared.supported_config_formats
                or iformat in self.shared.supported_stacking_formats
            ):

                create_file(self, id, iformat, file)

                # update saved file status
                index = self.shared.all_ids.index((id, hash, False))
                self.shared.all_ids[index] = (id, hash, True)
                message = "File uploaded!"

                # trigger index update for gui!
                self.shared.reindex_files()
            else:
                message = "Image format not supported!"
                status = False
        elif (id, hash, True) in self.shared.all_ids:
            message = "File with same name already exist!"
            status = False
        else:
            message = "Wrong ID or HASH!"
            status = False
        return message, status

    def upload_file_to_firebase(self, local_path: str, firebase_path: str) -> str:
        bucket = storage.bucket()
        blob = bucket.blob(firebase_path)
        blob.upload_from_filename(local_path)
        blob.make_public()  # Make the file public
        return blob.public_url

    def check_process_status(self, id: str, oformat: str) -> bool:
        baseUrl = "http://34.27.43.15:8000"
        response = requests.get(f'{baseUrl}/?func=processes')
        if response.status_code == 200:
            processes = response.json()
            for process in processes:
                if process['out'] == f"{id}{oformat}" and process['state'] in [3, 4]:
                    return True
        return False



    def createandtransform(
        self,
        id: str,
        hash: str,
        iformat: str,
        oformat: str,
        file: bytes,
        userId: str,
        *args,
        **kwargs
    ) -> Tuple[str, bool]:
        """
        Send image to server and start transform process
        @Return a tuple with a message and status
        @Return List[ response, status]
        """
        (message, status) = self.create(id=id, hash=hash, iformat=iformat, file=file)
        message += " "
        if status:
            message += Post(client=self.client, shared_variables=self.shared).transform(
                func="transform", id=id, oformat=oformat
            )
            # The message so far is:
            # "File uploaded! TransformProcess started! Query Process Status for more Information."
            # Define file paths
            image_local_path = f"/home/apps/forkedFloorplanToBlender3d/Server/storage/images/{id}{iformat}"
            obj_local_path = f"/home/apps/forkedFloorplanToBlender3d/Server/storage/objects/{id}{oformat}"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4())

            image_firebase_path = f"uploadedFloorplans/{userId}/{unique_id}-{timestamp}{iformat}"
            obj_firebase_path = f"convertedFloorplans/{userId}/{unique_id}{oformat}"

            # Upload image file to Firebase
            image_url = self.upload_file_to_firebase(image_local_path, image_firebase_path)
            message += f"\nImage uploaded to: {image_url}"

            # Polling for the .obj file readiness
            max_retries = 30  # Maximum number of retries
            retry_delay = 5  # Delay between retries in seconds

            for _ in range(max_retries):
                if self.check_process_status(id, oformat):
                    obj_url = self.upload_file_to_firebase(obj_local_path, obj_firebase_path)
                    message += f"\nOBJ uploaded to: {obj_url}"

                    # Add records to Firestore
                    dateTimeUploaded = datetime.now().isoformat()

                    # Define the new records
                    image_record = {
                        "dateTimeUploaded": dateTimeUploaded,
                        "path": image_firebase_path,
                        "successConversionTo3d": True,
                        "url": image_url
                    }
                    obj_record = {
                        "dateTimeUploaded": dateTimeUploaded,
                        "path": obj_firebase_path,
                        "type": "obj",
                        "url": obj_url
                    }

                    # Reference to the user document
                    user_ref = db.collection("user_floorplans").document(userId)

                    # Update the user document
                    user_ref.update({
                        "images": firestore.ArrayUnion([image_record]),
                        "objects": firestore.ArrayUnion([obj_record])
                    })

                    return message, True

                time.sleep(retry_delay)

            message += " OBJ file was not ready in time."
            return message, False

        return message, status
