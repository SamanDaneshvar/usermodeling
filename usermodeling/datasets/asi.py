"""Advanced Symbolics (ASI) dataset"""

import logging


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
