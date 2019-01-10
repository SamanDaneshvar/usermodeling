"""Perform deep learning on the datasets

This script trains a deep learning model on the datasets. %%
"""

import logging
import os
from datetime import datetime
import time
import sys

from keras.preprocessing.text import Tokenizer


def configure_root_logger():
    """Create a logger and set its configurations."""

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

    # • Make sure the *logs* folder is created inside the project directory, regardless of the current working directory
    script_path = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname or not)
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.
    # Package directory = the directory that the script file resides in
    # *os.path.dirname* goes one level up in the directory
    package_directory = os.path.dirname(script_path)
    project_directory = os.path.dirname(package_directory)
    LOGS_DIRECTORY = os.path.join(project_directory, "logs")
    # Create the directory if it does not exist
    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
    # Define the log file name
    LOG_FILE_NAME = datetime.today().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_file_path = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it to the handlers
    formatter = logging.Formatter("%(name)-16s: %(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def set_working_directory():
    """Log system info and set the current working directory

    This function logs current date and time, computer and user name, and script path.
    It also sets the current working directory = the project directory.
    """

    # Log current date and time, computer and user name, and script path
    logger.info("Current date and time: %s", datetime.today())
    logger.info("Computer and user name: %s, %s", os.getenv('COMPUTERNAME'), os.getlogin())
    # ↳ For a full list of environment variables and their values, call *os.environ*
    script_path = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname or not)
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.
    logger.info("Script path: %s", script_path)

    # • Set the project directory as the current working directory.
    # Package directory = the directory that the script file resides in
    # *os.path.dirname* goes one level up in the directory
    package_directory = os.path.dirname(script_path)
    #
    project_directory = os.path.dirname(package_directory)
    #
    if os.getcwd() == project_directory:
        logger.info("Current working directory = Project directory"
                    "\n")
    else:
        logger.info("Changing working directory from: %s", os.getcwd())
        # Change the working directory to the project directory
        os.chdir(project_directory)
        logger.info("Current working directory: %s"
                    "\n", os.getcwd())


def main():
    """The main function.

    Every time the script runs, it will call this function.
    """

    # Log run time
    logger.info("@ %.2f seconds: Run finished\n", time.process_time())


''' 
The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to "__main__" when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    logger = configure_root_logger()
    set_working_directory()
    main()
