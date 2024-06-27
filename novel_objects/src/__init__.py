import os


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    data_root_dir = root_dir()
    return os.path.split(data_root_dir)[0]
