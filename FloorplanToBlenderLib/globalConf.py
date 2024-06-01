import os
import random
import string
import logging
import json

from FloorplanToBlenderLib import const
# Initialize configuration

DEBUG_MODE = True
LOGGING_VERBOSE = True


def load_config_from_json(file_path):
    """
    Load configuration from a JSON file.
    @Param file_path: Path to the JSON file.
    @Return: Dictionary containing the configuration.
    """
    try:
        with open(file_path, 'r') as json_file:
            config = json.load(json_file)
        return config
    except Exception as e:
        logger = logging.getLogger('debug_logger')
        logger.error(f"Error loading configuration from {file_path}: {e}")
        return {}


def configure_logging():
    if LOGGING_VERBOSE:
        # Load the DEBUG_SESSION_ID from the JSON file
        debug_config = load_config_from_json('./config.json')
        
        log_dir_path = os.path.join('./storage/debug', debug_config['DEBUG_SESSION_ID'])
        log_file_path = os.path.join(log_dir_path, 'debug.log')
        os.makedirs(log_dir_path, exist_ok=True)

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

        # # Set the root logger's level to WARNING to avoid interference
        # logging.getLogger().setLevel(logging.WARNING)

        return logger
    return None





def generate_random_string(length=6):
    """
    Generate a random string of specified length.
    @Param length: Length of the random string.
    @Return: Random string.
    """
    return ''.join(random.choices(string.ascii_letters, k=length))

# def generate_random_number(length=12):
#     """
#     Generate a random number of specified length.
#     @Param length: Length of the random number.
#     @Return: Random number.
#     """
#     if length < 1:
#         raise ValueError("Length must be at least 1")
#     return str(int(''.join(random.choices('7893456012', k=length))))


# def initialize_debug_directory(session_id=None):
#     """
#     Initialize the debug directory for storing debug files.
#     @Param session_id: Unique identifier for the debug session.
#     @Return: Path to the debug directory.
#     """
#     logging.debug(f"Current session_id for debug directory: {session_id}")
#     # if not session_id:
#     #     session_id = generate_random_number()
#     debug_path = os.path.join('./storage/debug', session_id)
#     logging.debug(f"Initializing debug directory at: {debug_path}")
#     if not os.path.exists(debug_path):
#         os.makedirs(debug_path)
#         logging.debug(f"Debug directory created: {debug_path}")
#     return debug_path

# def update_config(debug_mode, logging_verbose, session_id):
#     """
#     Update configuration settings for debug mode and logging verbosity.
#     @Param debug_mode: Boolean to set debug mode.
#     @Param logging_verbose: Boolean to set logging verbosity.
#     @Param session_id: Unique identifier for the debug session.
#     """
#     global DEBUG_MODE, LOGGING_VERBOSE, DEBUG_SESSION_ID, DEBUG_STORAGE_PATH
#     DEBUG_MODE = debug_mode
#     LOGGING_VERBOSE = logging_verbose
#     DEBUG_SESSION_ID = session_id or '12345'
#     if DEBUG_MODE:
#         DEBUG_STORAGE_PATH = initialize_debug_directory(DEBUG_SESSION_ID)
#     else:
#         DEBUG_STORAGE_PATH = None
    
#     if LOGGING_VERBOSE:
#         logger.debug(f'Updated config: DEBUG_MODE={DEBUG_MODE}, LOGGING_VERBOSE={LOGGING_VERBOSE}, DEBUG_SESSION_ID={DEBUG_SESSION_ID}')


# Load the DEBUG_SESSION_ID from the JSON file
config = load_config_from_json('./config.json')
if 'DEBUG_SESSION_ID' not in config:
    raise ValueError("DEBUG_SESSION_ID not found in the configuration file")

DEBUG_SESSION_ID = config['DEBUG_SESSION_ID']

# DEBUG_STORAGE_PATH = initialize_debug_directory(DEBUG_SESSION_ID) if DEBUG_MODE else None
