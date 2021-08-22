from utils import get_image_paths


def preprocess_data(data_dir, N, M):
    image_paths = get_image_paths(data_dir)
    for img_path in image_paths:
        pass


if __name__ == "__main__":
    data_dir = "/home/idansheffer/data_others/professional_train_2020"
    N, M = 256, 256
    
    #         M
    #     ---------
    #    |         |
    #  N |         |
    #    |         |
    #     ---------
    
    preprocess_data(data_dir, N, M)