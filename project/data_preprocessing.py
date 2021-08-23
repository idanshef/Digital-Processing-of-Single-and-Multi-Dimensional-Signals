import cv2
import os
from utils import get_image_paths


def preprocess_data(data_dir, N, M):
    new_data_dir = f"{data_dir}_{N}_{M}"
    if not os.path.isdir(new_data_dir):
        os.makedirs(new_data_dir)
    
    image_paths = get_image_paths(data_dir)
    for img_path in image_paths:
        img = cv2.imread(img_path)
        rows, cols, _ = img.shape
        
        for row_block in range(rows // N):
            for col_block in range(cols // M):
                patch = img[row_block * N : row_block * N + N, col_block * M : col_block * M + M, :]
                file_name, file_extension = os.path.splitext(os.path.basename(img_path))
                patch_path = os.path.join(new_data_dir, f"{file_name}_{row_block}_{col_block}{file_extension}")
                if not os.path.isfile(patch_path):
                    cv2.imwrite(patch_path, patch)
                

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