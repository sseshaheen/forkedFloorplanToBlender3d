from typing import Tuple
from api.api import Api
from api.post import Post  # needed to call transform function
import base64

"""
FloorplanToBlender3d
Copyright (C) 2021 Daniel Westberg
"""


def create_file(ref, id, iformat, file):
    """Write incoming data to file"""
    # file_path = ref.shared.parentPath + "/" + ref.shared.imagesPath + "/" + id + iformat
    file_path = f"{ref.shared.parentPath}/{ref.shared.imagesPath}/{id}.{iformat}"
    # open(file_path, "wb").write(file)
    try:
        # Decode the base64 string
        if file.startswith("data:image/png;base64,"):
            file = file.replace("data:image/png;base64,", "")
        elif file.startswith("data:image/jpeg;base64,"):
            file = file.replace("data:image/jpeg;base64,", "")

        file_bytes = base64.b64decode(file)

        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logging.info(f"File successfully written to {file_path}")
        return True, f"File successfully written to {file_path}"
    except Exception as e:
        logging.error(f"Failed to write file: {str(e)}")
        return False, f"Failed to write file: {str(e)}"




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

                success, file_message = create_file(self, id, iformat, file)
                if success:
                    # Update saved file status
                    index = self.shared.all_ids.index((id, hash, False))
                    self.shared.all_ids[index] = (id, hash, True)
                    message = "File uploaded!"
                    # Trigger index update for GUI
                    self.shared.reindex_files()
                else:
                    message = file_message
                    status = False
            else:
                message = "Image format not supported!"
                status = False
        elif (id, hash, True) in self.shared.all_ids:
            message = "File with the same name already exists!"
            status = False
        else:
            message = "Wrong ID or HASH!"
            status = False
        
        return message, status

    def createandtransform(
        self,
        id: str,
        hash: str,
        iformat: str,
        oformat: str,
        file: bytes,
        *args,
        **kwargs
    ) -> Tuple[str, bool]:
        """
        Send image to server and start transform process
        @Return List[ response, status]
        """
        (message, status) = self.create(id=id, hash=hash, iformat=iformat, file=file)
        message += " "
        if status:
            message += Post(client=self.client, shared_variables=self.shared).transform(
                func="transform", id=id, oformat=oformat
            )
        return message, status
