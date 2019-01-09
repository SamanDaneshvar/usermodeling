"""Perform deep learning on the dataset

This script trains a deep learning model on the dataset. %%
"""

import logging
import os
from datetime import datetime
import time
import sys

from keras.preprocessing.text import Tokenizer


def configureRootLogger():
    """ This function creates a logger and sets its configurations.
    """

    # Create a RootLogger object
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    ''' ↳ The logger discards any logging calls with a level of severity lower than the level of the logger.
        Next, each handler decides to accept/discard the call based on its own level.
        By setting the level of the logger to NOTSET, we hand the power to handlers, and we don't filter out anything
        at the entrance. In effect, this is the same as setting the level to DEBUG (lowest level possible).
    '''

    # Create a console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # • Make sure the *logs* folder is created inside the project directory, regardless of the current working directory
    scriptPath = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname or not)
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.
    # Package directory = the directory that the script file resides in
    # *os.path.dirname* goes one level up in the directory
    packageDirectory = os.path.dirname(scriptPath)
    projectDirectory = os.path.dirname(packageDirectory)
    LOGS_DIRECTORY = os.path.join(projectDirectory, "logs")
    # Create the directory if it does not exist
    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
    # Define the log file name
    LOG_FILE_NAME = datetime.today().strftime("%Y-%m-%d_%H-%M-%S.log")
    logFilePath = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

    # Create a file handler
    fileHandler = logging.FileHandler(logFilePath, encoding="utf-8")
    fileHandler.setLevel(logging.DEBUG)

    # Create a formatter and set it to the handlers
    formatter = logging.Formatter("%(name)-16s: %(levelname)-8s %(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


def checkSystemInfo():
    """This function logs current date and time, computer and user name, and script path.
    It also ensures that the current working directory is the same as the script directory.
    """

    # Log current date and time, computer and user name, and script path
    logger.info("Current date and time: %s", datetime.today())
    logger.info("Computer and user name: %s, %s", os.getenv('COMPUTERNAME'), os.getlogin())
    # ↳ For a full list of environment variables and their values, call *os.environ*
    scriptPath = os.path.realpath(sys.argv[0])
    # ↳ *sys.argv[0]* contains the script name (it is operating system dependent whether this is a full pathname or not)
    # ↳ *realpath* eliminates symbolic links and returns the canonical path.
    logger.info("Script path: %s", scriptPath)

    # • Set the project directory as the current working directory.
    # Package directory = the directory that the script file resides in
    # *os.path.dirname* goes one level up in the directory
    packageDirectory = os.path.dirname(scriptPath)
    #
    projectDirectory = os.path.dirname(packageDirectory)
    #
    if os.getcwd() == projectDirectory:
        logger.info("Current working directory = Project directory"
                    "\n")
    else:
        logger.info("Changing working directory from: %s", os.getcwd())
        # Change the working directory to the project directory
        os.chdir(projectDirectory)
        logger.info("Current working directory: %s"
                    "\n", os.getcwd())


def main():
    """This function is the “main” function.
    Every time the script runs, it will call this function.
    """

    logger.info("This is the main function!")

    # Log run time
    logger.info("@ %.2f seconds: Run finished\n", time.process_time())


''' The following lines will be executed only if this .py file is run as a script,
and not if it is imported as a module.
• __name__ is one of the import-related module attributes, which holds the name of the module.
• A module's __name__ is set to "__main__" when it is running in
the main scope (the scope in which top-level code executes).  
'''
if __name__ == "__main__":
    logger = configureRootLogger()
    checkSystemInfo()
    main()
