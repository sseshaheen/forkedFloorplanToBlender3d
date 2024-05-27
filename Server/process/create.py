from datetime import datetime 
"""
FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""
"""
The process class represents a thread handling stuff in new threads
"""
from file.file_handler import FileHandler
from process.process import Process
from subprocess import check_output
import os
import sys
import logging

sys.path.insert(0, "..")
from FloorplanToBlenderLib import (
    config,
    floorplan,
    generate,
    execution,
    IO,
    const,
    globalConf,
)  # floorplan to blender lib

"""This process should create a 3d object file using the FTBLibrary"""


class Create(Process):
    def __init__(self, func, id, oformat, shared_variables):
        super().__init__(shared_variables=shared_variables)
        # We expect following fields in data
        # {'func': 'transform', 'id':id, 'format':'.obj'}
        self.process["task"] = func
        self.process["in"] = id
        self.process["cstate"] = 4  # set amount of state -1 here, useful for gui later!
        # TODO: check if "format field exist!"
        self.process["format"] = oformat
        # we will overwrite old objects!
        self.update("out", id + oformat)

        # self.process["initiation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.process["create_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    def run(self):
        # This is where the new thread will start
        image_path = self.shared.get_file_path(
            self.process["in"], self.shared.imagesPath, self.shared.images
        )

        if image_path is None:
            self.process["state"] = -1
            self.update("status", "ERROR: image with id not found.")
            return

        # TODO evaluate if wanted
        # TODO removenoice
        # TODO resize if wanted

        # hax fix for now
        const.BASE_PATH = f"./storage/data/{self.process['in']}/"
        const.DOOR_MODEL = "../Images/Models/Doors/door.png"
        const.DEFAULT_CALIBRATION_IMAGE_PATH = (
            "../Images/Calibrations/wallcalibration.png"
        )
        blender_install_path = IO.blender_installed()
        config.get_default_blender_installation_path()

        # Set other paths (don't need to change these)
        program_path = os.getcwd()

        blender_script_path = "../Blender/floorplan_to_3dObject_in_blender.py"

        # print(program_path, blender_script_path)

        IO.clean_data_folder(f"./storage/data/{self.process['in']}")

        # Remove target file if it already exists!
        # Else we will get a bad rename!
        fh = FileHandler()
        tmp = f"./storage/objects/{self.process['in']}.blend"
        if os.path.isfile(tmp):
            fh.remove(tmp)

        self.process["state"] = self.process["state"] + 1
        self.update("status", "Image processing calculations")

        # Generate data files
        # TODO: fix this to work for several instances at once!
        generate.base_path = f"./storage/data/{self.process['in']}"
        generate.path = f"./storage/data/{self.process['in']}"
        target_path = f"./storage/objects/{self.process['in']}.blend"
        config_path = None
        f = floorplan.new_floorplan(config_path)
        f.image_path = image_path

        data_paths = list()
        data_paths = [execution.simple_single(f, False)]
        # Debug print
        """
        print(str([blender_install_path,
        "-noaudio", # this is an ubuntu hax fix
        "--background",
        "--python",
        blender_script_path,
        program_path # Send this as parameter to script
        ] +  data_paths))
        """
        self.process["state"] = self.process["state"] + 1
        self.update("status", "Creating objects in Blender3d")

        # Create blender project
        # TODO: change script to decide format!
        check_output(
            [
                blender_install_path,
                "-noaudio",  # this is a dockerfile ubuntu hax fix
                "--background",
                "--python",
                blender_script_path,  # Send this as parameter to script
                program_path + "/",
                target_path,
            ]
            + data_paths
        )

        self.process["state"] = self.process["state"] + 1
        self.update("status", "Create Object file")

        check_output(
            [
                blender_install_path,
                "-noaudio",  # this is a dockerfile ubuntu hax fix
                "--background",
                "--python",
                "../Blender/blender_export_any.py",
                f"./storage/objects/{self.process['in']}.blend",
                self.process["format"],
                f"./storage/objects/{self.process['out']}",
            ]
        )

        self.process["state"] = self.process["state"] + 1
        self.update("status", "Cleanup")

        # Ensure the correct directory is targeted for removal
        work_dir = f"./storage/data/{self.process['in']}/0/"
        if os.path.isdir(work_dir):
            fh.remove(work_dir)
        else:
            # Handle the case where files might be misnamed
            files = [f for f in os.listdir(f"./storage/data/{self.process['in']}") if f.startswith("0")]
            for f in files:
                file_path = f"./storage/data/{self.process['in']}/{f}"
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    fh.remove(file_path)

        self.process["state"] = self.process["state"] + 1
        self.update("status", "Done")

        # Reindex here
        self.shared.reindex_files()
