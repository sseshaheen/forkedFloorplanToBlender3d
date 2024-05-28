import os
import random
import string
import logging
from FloorplanToBlenderLib import const
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_random_string(length=6):
    """
    Generate a random string of specified length.
    @Param length: Length of the random string.
    @Return: Random string.
    """
    return ''.join(random.choices(string.ascii_letters, k=length))

def generate_random_number(length=12):
    """
    Generate a random number of specified length.
    @Param length: Length of the random number.
    @Return: Random number.
    """
    if length < 1:
        raise ValueError("Length must be at least 1")
    return str(int(''.join(random.choices('7893456012', k=length))))


def initialize_debug_directory(session_id=None):
    """
    Initialize the debug directory for storing debug files.
    @Param session_id: Unique identifier for the debug session.
    @Return: Path to the debug directory.
    """
    logging.debug(f"Current session_id for debug directory: {session_id}")
    if not session_id:
        session_id = generate_random_number()
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
    DEBUG_SESSION_ID = session_id or '12345'
    if DEBUG_MODE:
        DEBUG_STORAGE_PATH = initialize_debug_directory(DEBUG_SESSION_ID)
    else:
        DEBUG_STORAGE_PATH = None
    
    if LOGGING_VERBOSE:
        logger.debug(f'Updated config: DEBUG_MODE={DEBUG_MODE}, LOGGING_VERBOSE={LOGGING_VERBOSE}, DEBUG_SESSION_ID={DEBUG_SESSION_ID}')

# Initialize configuration

DEBUG_MODE = True
LOGGING_VERBOSE = True

DEBUG_SESSION_ID = '987654321'
DEBUG_STORAGE_PATH = initialize_debug_directory(DEBUG_SESSION_ID) if DEBUG_MODE else None