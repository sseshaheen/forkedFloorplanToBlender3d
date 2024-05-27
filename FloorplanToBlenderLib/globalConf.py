import os
import random
import string
import logging
from FloorplanToBlenderLib import const
# Configure logging
logger = logging.getLogger(__name__)

def generate_random_string(length=6):
    """
    Generate a random string of specified length.
    @Param length: Length of the random string.
    @Return: Random string.
    """
    return ''.join(random.choices(string.ascii_letters, k=length))

# Set debug mode (True for debug mode, False for normal mode)
DEBUG_MODE = False

# Set logging verbosity (True for detailed logging, False for concise logging)
LOGGING_VERBOSE = False

# Generate a unique identifier for the debug session
DEBUG_SESSION_ID = generate_random_string()

# Define the storage path for debug images
# DEBUG_STORAGE_PATH = const.BASE_PATH = f"./storage/data/debug/{DEBUG_SESSION_ID}"
DEBUG_STORAGE_PATH = const.BASE_PATH = "/home/apps/forkedFloorplanToBlender3d/Server/storage/data/debug/{DEBUG_SESSION_ID}"

def initialize_debug_directory(session_id):
    """
    Initialize the debug directory for storing debug files.
    @Param session_id: Unique identifier for the debug session.
    @Return: Path to the debug directory.
    """
    debug_path = os.path.join('./storage/debug', session_id)
    logging.debug(f"Initializing debug directory at: {debug_path}")
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
        logging.debug(f"Debug directory created: {debug_path}")
    return debug_path

def update_config(debug_mode, logging_verbose, session_id):
    """
    Update configuration settings for debug mode and logging verbosity.
    @Param debug_mode: Boolean to set debug mode.
    @Param logging_verbose: Boolean to set logging verbosity.
    @Param session_id: Unique identifier for the debug session.
    """
    global DEBUG_MODE, LOGGING_VERBOSE, DEBUG_SESSION_ID, DEBUG_STORAGE_PATH
    DEBUG_MODE = debug_mode
    LOGGING_VERBOSE = logging_verbose
    DEBUG_SESSION_ID = session_id or generate_random_string()
    if DEBUG_MODE:
        DEBUG_STORAGE_PATH = initialize_debug_directory(DEBUG_SESSION_ID)
    
    if LOGGING_VERBOSE:
        logger.debug(f'Updated config: DEBUG_MODE={DEBUG_MODE}, LOGGING_VERBOSE={LOGGING_VERBOSE}, DEBUG_SESSION_ID={DEBUG_SESSION_ID}')

# Initialize the debug directory upon module import if debug mode is enabled
if DEBUG_MODE:
    DEBUG_STORAGE_PATH = initialize_debug_directory(DEBUG_SESSION_ID)
