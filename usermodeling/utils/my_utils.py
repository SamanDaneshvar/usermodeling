"""Utilities"""

from datetime import datetime
import hashlib
import logging
import os
import pickle
import sys


def configure_root_logger(proj_dir=1):
    """Create a logger and set its configurations.

    Args:
        proj_dir: An integer or a string, indicating the project directory.
            Similar to the *set_working_directory()* function. Refer to that function's docstrings for more details.

    Returns:
        logger: The logger object
        RUN_TIMESTAMP: A string containing the date and time of the run with the format '%Y-%m-%d_%H-%M-%S'
            This can be used when naming other output files, so that they match the log file.
    """

    # Create a RootLogger object
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    ''' 
    ↳ The logger discards any logging calls with a level of severity lower than the level of the logger.
    Next, each handler decides to accept/discard the call based on its own level.
    By setting the level of the logger to NOTSET, we hand the power to handlers, and we don't filter out anything
    at the entrance. In effect, this is the same as setting the level to DEBUG (lowest level possible).
    '''

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # • Get the script path and script (module) name.
    SCRIPT_PATH = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname or not)
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.
    SCRIPT_FILENAME = os.path.basename(SCRIPT_PATH)
    # Trim the '.py' extension to get the name of the script (module). If the script filename does not have a '.py'
    # extention, don't trim anything.
    if SCRIPT_FILENAME[-3:] == '.py':
        SCRIPT_NAME = SCRIPT_FILENAME[:-3]
    else:
        SCRIPT_NAME = SCRIPT_FILENAME

    # Deduce the project directory.
    # We want to make sure the *logs* folder is created inside the project directory, regardless of
    # the current working directory.
    PROJECT_DIRECTORY = deduce_project_directory(proj_dir, SCRIPT_PATH)

    # Assemble the logs directory
    LOGS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'logs')
    # Create the directory if it does not exist
    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
    # Create the run timestamp. This can also be used later for output filenames.
    RUN_TIMESTAMP = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    # Assemble the log file name
    LOG_FILE_NAME = RUN_TIMESTAMP + ' ' + SCRIPT_NAME + '.log'
    LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

    # Create a file handler
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it to the handlers
    formatter = logging.Formatter('%(name)-32s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, RUN_TIMESTAMP


def configure_basic_logger():
    """Create a basic logger with a console handler (without any file handlers) and set its configurations.

    Returns:
        logger: The logger object
    """

    # Create a RootLogger object
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it to the handler
    formatter = logging.Formatter('%(name)-32s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


def set_working_directory(proj_dir=1):
    """Log system info and set the current working directory

    This function logs current date and time, computer and user name, and script path.
    It also sets the working directory. The goal is to set the project directory as the working directory.

    Args:
        proj_dir: An integer or a string, indicating the project directory (new working directory).
            - When str: Contains the absolute path of the project directory (new working directory).
            - When int: Indicates that the project directory (new working directory) is this many levels higher than
                the script directory (the directory where the script that is running in the main scope resides).

    Raises:
        TypeError: If the input is not of type int or str.
    """

    # Log current date and time, computer and user name
    logger.info('Current date and time:  %s', datetime.today())
    logger.info('Computer and user name: %s, %s', os.getenv('COMPUTERNAME'), os.getlogin())
    # ↳ For a full list of environment variables and their values, call *os.environ*

    # Get the script path (of the script that is running in the main scope)
    script_path = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname
    # or not) of the script that is running in the main scope (the scope in which top-level code executes).
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.

    # Log the script path
    logger.info('Main scope script path:          %s', script_path)

    # Deduce the absolute project directory.
    project_directory = deduce_project_directory(proj_dir, script_path)

    # Set the current working directory = project directory
    if os.getcwd() == project_directory:
        logger.info('Current working directory = Project directory')
    else:
        logger.info('Changing working directory from: %s', os.getcwd())
        # Change the working directory to the project directory
        os.chdir(project_directory)

    logger.info('Current working directory:       %s'
                '\n', os.getcwd())


def deduce_project_directory(proj_dir, script_path):
    """Deduce the absolute project directory

    This is a helper function for *set_working_directory()*. Refer to that function's docstrings for more details.
    """

    # If *proj_dir* is a string, it is expected to already contain the absolute project directory.
    # If *proj_dir* is of type *int*, go this many levels higher from the script directory to deduce the
    # absolute project directory (*project_directory*).
    if isinstance(proj_dir, str):
        project_directory = proj_dir
    elif isinstance(proj_dir, int):
        # Start with the script directory
        # *os.path.dirname* goes one level up in the directory
        directory = os.path.dirname(script_path)
        # Go *proj_dir* levels up
        for i in range(proj_dir):
            directory = os.path.dirname(directory)
        project_directory = directory
        # *project_directory* now contains the absolute project directory.
    else:
        raise (TypeError('The input is expected to be of type int or str, not %s' % type(proj_dir)))

    return project_directory


def hex_hash_object(input_object):
    """Generate the SHA1 digest (hash value) of an object as a hexadecimal string.

    This is useful for comparing two objects or one object in different runs.

    Args:
        input_object: An object with any type
    Returns:
        The SHA1 digest (hash value) of the *input_object* as a string, containing 40 hexadecimal digits.
    """

    # Convert the input object to a *bytes* object (the pickled representation of the input object as a *bytes* object)
    input_object_as_bytes = pickle.dumps(input_object)
    # ↳ An inferior alternative could be *str(input_object).encode("utf-8")*

    # Create a hash object that uses the SHA1 algorithm
    hash_object = hashlib.sha1()

    # Update the hash object with the *bytes* object. This will calculate the hash value.
    hash_object.update(input_object_as_bytes)

    # Get the hexadecimal digest (hash value)
    hex_hash_value = hash_object.hexdigest()

    return hex_hash_value


'''
The following lines will be executed any time this .py file is run as a script or imported as a module.
'''
# Create a logger object. The root logger would be the parent of this logger
# Note that if you run this .py file as a script, this logger will not function, because it is not configured.
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # The following lines will be executed only if this .py file is run as a script,
    # and not if it is imported as a module.
    print("Module was executed directly.")
