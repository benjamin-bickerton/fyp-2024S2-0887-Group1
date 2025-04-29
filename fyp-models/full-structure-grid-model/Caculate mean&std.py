import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms

def compute_mean_std(image_dir):
    transform = transforms.ToTensor()  # Converts to [C, H, W], range [0,1]
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, fname))

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)  # [3, H, W]
        num_pixels += tensor.shape[1] * tensor.shape[2]
        pixel_sum += tensor.sum(dim=[1, 2])
        pixel_squared_sum += (tensor ** 2).sum(dim=[1, 2])

    mean = pixel_sum / num_pixels
    std = (pixel_squared_sum / num_pixels - mean ** 2).sqrt()

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    img_folder = "C:\\Users\Administrator\OneDrive - Monash University\桌面\Ben\\new\dataset\\all"
    mean, std = compute_mean_std(img_folder)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
