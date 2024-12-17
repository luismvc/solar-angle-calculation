import argparse
import sys

from src.img_proc import process
from src.utils import get_param_file, log


def main(config_file_: dict) -> int:
    """Main function"""
    log.info("Image prcessing")

    process(config_file)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--configFile", help="configuration file")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        log.error(
            "Configuration file missing\n  usage:\n\t python <main>.py -conf <path to conf file>/<config_file_name>.yaml\n"
        )

        sys.exit(-1)

    config_file = get_param_file(args)
    main(config_file)
