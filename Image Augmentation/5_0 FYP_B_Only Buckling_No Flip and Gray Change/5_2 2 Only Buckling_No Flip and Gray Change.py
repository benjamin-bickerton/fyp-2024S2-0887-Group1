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

def augment_image(image_np, num_augments, rotate, gaussian):
    seq = iaa.Sequential([
        iaa.Affine(rotate=rotate),
        iaa.GaussianBlur(sigma=gaussian)
    ])
    return [seq(image=image_np) for _ in range(num_augments)]

def process_single_folder(input_subfolder, output_subfolder, parent_folder_name,
                          num_augments, scale_factor, rotate, gaussian):
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    for filename in os.listdir(input_subfolder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_subfolder, filename)
            image = Image.open(file_path).convert('RGBA')
            image_np = np.array(image)

            processed_image = preprocess_image(image_np, scale_factor)
            augmented_images = augment_image(processed_image, num_augments, rotate, gaussian)

            for idx, aug_image in enumerate(augmented_images):
                output_filename = f'{parent_folder_name}_aug_{idx+1}.png'
                output_path = os.path.join(output_subfolder, output_filename)
                Image.fromarray(aug_image).save(output_path)

def process_all_folders(input_folder, output_folder, num_augments, scale_factor, rotate, gaussian):
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(output_folder, subfolder_name)
            process_single_folder(
                input_subfolder=subfolder_path,
                output_subfolder=output_subfolder,
                parent_folder_name=subfolder_name,  # 把母文件夹名传进去
                num_augments=num_augments,
                scale_factor=scale_factor,
                rotate=rotate,
                gaussian=gaussian
            )

# =================== Adjustable Parameters ===================

input_folder = r'C:\Users\20818\OneDrive\Desktop\CIV 4701&4702\More Dataset\FYP_B\Update_Aug_Buckling Dataset\9'
output_folder = r'C:\Users\20818\OneDrive\Desktop\CIV 4701&4702\More Dataset\FYP_B\Update_Aug_Buckling Dataset\9\9_Aug'

num_augments = 100
scale_factor = 0.85
rotate = (-25, 25)
gaussian = (0, 3.0)

# =================== Run ===================
if __name__ == "__main__":
    process_all_folders(
        input_folder=input_folder,
        output_folder=output_folder,
        num_augments=num_augments,
        scale_factor=scale_factor,
        rotate=rotate,
        gaussian=gaussian
    )
