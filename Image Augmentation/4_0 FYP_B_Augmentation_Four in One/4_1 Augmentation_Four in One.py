import os
import cv2
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa

def preprocess_image(image_np, scale_factor):
    new_height, new_width = int(image_np.shape[0] * scale_factor), int(image_np.shape[1] * scale_factor)
    resized_image = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_height = (image_np.shape[0] - new_height) // 2
    pad_width = (image_np.shape[1] - new_width) // 2
    padded_image = cv2.copyMakeBorder(
        resized_image, pad_height, pad_height, pad_width, pad_width,
        cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )
    return padded_image

def augment_image(image_np, num_augments, flip_lr, flip_ud, rotate, gaussian):
    seq = iaa.Sequential([
        iaa.Fliplr(flip_lr),
        iaa.Flipud(flip_ud),
        iaa.Affine(rotate=rotate),
        iaa.GaussianBlur(sigma=gaussian)
    ])
    return [seq(image=image_np) for _ in range(num_augments)]

def merge_images(images, final_size=(640, 640)):
    img_h, img_w = final_size[0] // 2, final_size[1] // 2
    resized = [cv2.resize(img, (img_w, img_h)) for img in images]
    top = np.concatenate((resized[0], resized[1]), axis=1)
    bottom = np.concatenate((resized[2], resized[3]), axis=1)
    merged = np.concatenate((top, bottom), axis=0)
    return merged

def process_all_folders(parent_input_folder, parent_output_folder, num_augments, scale_factor,
                        flip_lr, flip_ud, rotate, gaussian):

    for folder_name in os.listdir(parent_input_folder):
        folder_path = os.path.join(parent_input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        output_path = os.path.join(parent_output_folder, folder_name)
        os.makedirs(output_path, exist_ok=True)

        categories = {'A': [], 'S': [], 'ZB': [], 'YB': []}
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith('.png'):
                for key in categories:
                    if file.startswith(key):
                        categories[key].append(file_path)

        min_len = min(len(v) for v in categories.values())
        for i in range(min_len):
            originals = []
            for key in ['A', 'S', 'ZB', 'YB']:
                img = Image.open(categories[key][i]).convert('RGBA')
                img_np = np.array(img)
                img_pre = preprocess_image(img_np, scale_factor)
                originals.append(img_pre)

            for j in range(num_augments):
                aug_set = [augment_image(img, 1, flip_lr, flip_ud, rotate, gaussian)[0] for img in originals]
                merged = merge_images(aug_set)
                save_path = os.path.join(output_path, f"{folder_name}_{i}_{j+1}.png")
                Image.fromarray(merged).save(save_path)

# ========== 可调参数 ==========

parent_input_folder = r'C:\Users\20818\OneDrive\Desktop\CIV 4701&4702\More Dataset\FYP_B\Augmentation\Dataset_Ground Floor\9\9_Input'
parent_output_folder = r'C:\Users\20818\OneDrive\Desktop\CIV 4701&4702\More Dataset\FYP_B\Augmentation\Dataset_Ground Floor\9\9_Ground Floor Augmentation Dataset'

if __name__ == "__main__":
    process_all_folders(
        parent_input_folder=parent_input_folder,
        parent_output_folder=parent_output_folder,
        num_augments=100,
        scale_factor=0.85,
        flip_lr=0.5,
        flip_ud=0.2,
        rotate=(-25, 25),
        gaussian=(0, 3.0)
    )
