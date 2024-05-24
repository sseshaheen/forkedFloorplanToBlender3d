from shared_variables import shared_variables
from api.api import Api
from process.create import Create
from file.file_handler import FileHandler
import logging

"""
FloorplanToBlender3d
Copyright (C) 2021 Daniel Westberg
"""


class Post(Api):
    def __init__(self, client, shared_variables):
        super().__init__(client, shared_variables)
        # All all viable functions here!
        self.dispatched_calls["create"] = self.create
        self.dispatched_calls["remove"] = self.remove
        self.dispatched_calls["transform"] = self.transform

    def create(self, *args, **kwargs) -> str:
        """
        Create new specific file name/id and hash, return JSON.
        """
        tmp_id = None

        while self.shared.id_exist(tmp_id) or tmp_id is None:
            tmp_id = self.shared.id_generator()

        pair = (tmp_id, self.shared.hash_generator(tmp_id), False)
        self.shared.all_ids.append(pair)

        return str(pair)

    def remove(self, id: str, *args, **kwargs) -> str:
        """Remove all files linked to specified id."""
        logging.info(f"Received remove request for id: {id}")
        try:
            fs = FileHandler()
            # This will remove all files related to id
            for file in self.shared.all_files:
                if id in file:
                    for f in self.shared.get_id_files(id):
                        logging.info(f"Removing file: {f}")
                        fs.remove(f)
            self.shared.reindex_files()
            logging.info(f"Successfully removed files for id: {id}")
            return "Removed id!"
        except Exception as e:
            logging.error(f"Error removing files for id {id}: {e}")
            return f"Error removing files for id {id}: {e}"

    def transform(self, func: str, id: str, oformat: str, *args, **kwargs) -> str:
        """Transform Image to 3dObject."""
        _id = self.shared.get_id(id)
        if _id is not None and _id[2]:
            if oformat in self.shared.supported_blender_formats:
                Create(
                    func=func, id=id, oformat=oformat, shared_variables=self.shared
                ).start()
                message = "TransformProcess started! Query Process Status for more Information."
            else:
                message = "Format not supported!"
        else:
            message = "File doesn't exist!"
            self.shared.bad_client_event(self.client)
        return message
