import os
import sys
import logging
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CNN.")
    parser.add_argument(
        "--model",
        help="model name to benchmark",
        default=0,
        type=str,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()



def setup_logger(save_dir):
    logger = logging.getLogger('cnnbenchmark.')
    logger.setLevel(logging.DEBUG)
    logger.propogate = False
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    filename = os.path.join(save_dir, '{}.log'.format(datetime.now()))
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger('cnnbenchmark.' + name)
