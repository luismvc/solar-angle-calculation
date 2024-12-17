import logging
import re
import sys
from os import listdir
from os.path import isfile, join

import yaml

LOG_LEVEL = logging.DEBUG
LOGFORMAT = (
    # "  %(log_color)s%(levelname)-4s%(reset)s | %(log_color)s%(message)s%(reset)s"
    "  %(log_color)s%(message)s%(reset)s"
)
from colorlog import ColoredFormatter

logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger("pythonConfig")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)


def atoi(text):
    return int(text) if text.isdigit() else text


def get_param_file(args):
    """This function reads an input argument specifying the configuration file and extracts the data from it."""

    import os.path

    if os.path.exists(args.configFile):
        with open(args.configFile) as file:
            config_param = yaml.load(file, Loader=yaml.FullLoader)

    else:
        log.error("Configuration file doesn't exist")
        sys.exit(-1)

    return config_param


def load_file_names_data(mypath):
    log.info("Loading data file names...")
    log.info(mypath)

    # import os.path
    # For Sorting
    import re
    from os import path

    def natural_keys(text):
        """
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        """
        return [atoi(c) for c in re.split(r"(\d+)", text)]

    if not path.exists(mypath):
        False, []

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # Sorting file names
    onlyfiles.sort(key=natural_keys)

    n_files = len(onlyfiles)

    # if __debug__:
    #     log.debug("--- files ---")
    #     print("Number of files: ",len(onlyfiles))
    #     # print("\nFile names")
    #     # print(onlyfiles)
    #     # print("\n")
    log.info("\tNumber of data file names loaded: {}".format(n_files))

    return True, onlyfiles


def load_data(file_path):

    flag_loaded, fileNames = load_file_names_data(file_path)

    if flag_loaded:
        return fileNames
    return None


def split_string(s):
    # Split the string by both hyphens and underscores
    return re.split("[-_.]", s)
