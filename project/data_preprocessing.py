import cv2
import os
from glob import glob


def preprocess_data(data_dir, patch_height, patch_width):
    new_data_dir = f"{data_dir}_{patch_height}_{patch_width}"
    if not os.path.isdir(new_data_dir):
        os.makedirs(new_data_dir)
    
    image_paths = glob(os.path.join(data_dir, "*"))
    n_images = len(image_paths)
    for idx, img_path in enumerate(image_paths):
        print(f"Decomposing image {idx + 1}/{n_images}")
        img = cv2.imread(img_path)
        rows, cols, _ = img.shape
        
        for row_block in range(rows // patch_height):
            for col_block in range(cols // patch_width):
                patch = img[row_block * patch_height : (row_block + 1) * patch_height,
                            col_block * patch_width : (col_block + 1) * patch_width, :]
                file_name, file_extension = os.path.splitext(os.path.basename(img_path))
                patch_path = os.path.join(new_data_dir, f"{file_name}_{row_block}_{col_block}{file_extension}")
                if not os.path.isfile(patch_path):
                    cv2.imwrite(patch_path, patch)
                

if __name__ == "__main__":
    data_dir = "/home/orweiser/university/Digital-Processing-of-Single-and-Multi-Dimensional-Signals/data/valid"
    N, M = 256, 256
    preprocess_data(data_dir, N, M)