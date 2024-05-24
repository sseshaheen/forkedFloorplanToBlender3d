from typing import Tuple
from api.api import Api
from api.post import Post  # needed to call transform function
import firebase_admin
from firebase_admin import credentials, storage
import os
import uuid
from datetime import datetime

# Path to the downloaded service account key
# TODO: verify incoming userId through appropriate authentcation received from the mobile app
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
    ) -> Dict[str, any]:
        """
        Send image to server and start transform process
        @Return a dictionary with response details
        """
        (message, status) = self.create(id=id, hash=hash, iformat=iformat, file=file)
        response = {"message": message, "status": status, "success": status}
        
        if status:
            transform_message = Post(client=self.client, shared_variables=self.shared).transform(
                func="transform", id=id, oformat=oformat
            )
            response["message"] += f" {transform_message}"

            # Define file paths
            image_local_path = f"/home/apps/forkerFloorplanToBlender3d/Server/storage/images/{id}{iformat}"
            obj_local_path = f"/home/apps/forkerFloorplanToBlender3d/Server/storage/objects/{id}{oformat}"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4())

            image_firebase_path = f"uploadedFloorplans/{userId}/{unique_id}-{timestamp}{iformat}"
            obj_firebase_path = f"convertedFloorplans/{userId}/{unique_id}.obj"

            try:
                # Upload files to Firebase
                image_url = self.upload_file_to_firebase(image_local_path, image_firebase_path)
                obj_url = self.upload_file_to_firebase(obj_local_path, obj_firebase_path)

                # Add URLs to the response
                response["image_url"] = image_url
                response["obj_url"] = obj_url
            except Exception as e:
                response["message"] += f" Error uploading files: {str(e)}"
                response["status"] = False
                response["success"] = False

        return response

