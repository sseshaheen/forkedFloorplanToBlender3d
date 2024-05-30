import json
import logging
from . import const
from . import config
from .globalConf import load_config_from_json, DEBUG_MODE, LOGGING_VERBOSE
import os
"""
Floorplan
This file contains functions for handling the floorplan class.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""
def configure_logging():
    if LOGGING_VERBOSE:
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')
        
        log_dir_path = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'])
        log_file_path = os.path.join(log_dir_path, 'debug.log')
        os.makedirs(os.path.dirname(log_dir_path), exist_ok=True)

        # Create a logger
        logger = logging.getLogger('debug_logger')
        logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

        # Create a file handler to log everything
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a console handler to log warnings and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    return None




def save_debug_info(filename, data):
    """
    Save debug information to a file if DEBUG_MODE is enabled.
    """
    if DEBUG_MODE:
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')

        DEBUG_STORAGE_PATH = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'], 'txt')
        if not os.path.exists(DEBUG_STORAGE_PATH):
            os.makedirs(DEBUG_STORAGE_PATH)
        filepath = os.path.join(DEBUG_STORAGE_PATH, filename)
        with open(filepath, 'a') as file:
            file.write(json.dumps(data, indent=4))
        # if LOGGING_VERBOSE:
        #     logger = configure_logging()
        #     if logger:
        #         logger.debug(f'Saved debug info: {filepath}')



def new_floorplan(config):
    """
    Creates and returns a new floorplan class from a config file.
    @Param config: Path to the config file.
    @Return: New floorplan object.
    """
    return Floorplan(config)

class Floorplan:
    """
    This class represents a floorplan.
    It simplifies the code and enhances customizability.
    """

    def __init__(self, conf=None):
        """
        Initialize the floorplan with the given configuration.
        @Param conf: Path to the config file.
        """
        if conf is None:
            # Use default config file if none provided
            conf = const.IMAGE_DEFAULT_CONFIG_FILE_NAME
        self.conf = conf
        self.create_variables_from_config(self.conf)
        
        if LOGGING_VERBOSE:
            logger = configure_logging()
            if logger:
                logger.debug('Initialized Floorplan with config.')
        save_debug_info('floorplan_init.txt', {'config': self.conf, 'variables': vars(self)})

    def __str__(self):
        """
        String representation of the floorplan object.
        @Return: String representation of the floorplan variables.
        """
        return str(vars(self))

    def create_variables_from_config(self, conf):
        """
        Create variables for the floorplan object from the configuration file.
        @Param conf: Path to the config file.
        """
        settings = config.get_all(conf)
        settings_dict = {s: dict(settings.items(s)) for s in settings.sections()}
        for group in settings_dict.items():  # Ignore group names
            for item in group[1].items():
                setattr(self, item[0], json.loads(item[1]))

        if LOGGING_VERBOSE:
            logger = configure_logging()
            if logger:
                logger.debug('Created variables from config.')
        save_debug_info('create_variables_from_config.txt', {'config': conf, 'variables': vars(self)})

        # Debug
        # print(vars(self))
