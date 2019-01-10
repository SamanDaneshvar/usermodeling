"""Perform deep learning on the datasets

This script trains a deep learning model on the datasets. %%
"""

from usermodeling import utils

import time

from keras.preprocessing.text import Tokenizer


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
    logger = utils.configure_root_logger()
    utils.set_working_directory()
    main()
