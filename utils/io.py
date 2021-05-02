import os

def check_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

