import os


def get_image_paths(data_dir):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        image_paths = [os.path.join(root, x) for x in files]
    return image_paths