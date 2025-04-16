import os
import csv
import torch
import numpy as np
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        loss = output[:, class_idx]
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

def compute_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return accuracy, precision, recall, f1

def overlay_cam_on_image(img_tensor, cam_tensor):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.44, 43, 0.44]) + np.array([0.61, 0.66, 0.62]))  # unnormalize
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    cam = cam_tensor.squeeze().cpu().numpy()
    cam = (cam * 255).astype(np.uint8)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.5, cam, 0.5, 0)
    return overlay

def test_model(test_dir, model_path="./save_weights/best_model.pth", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "originals"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "overlays"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.61, 0.66, 0.62],
                             [0.44, 0.43, 0.44])
    ])

    raw_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    raw_dataset = datasets.ImageFolder(test_dir, transform=raw_transform)  # For raw images
    class_names = dataset.classes
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False)

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model, target_layer=model.features[-1])

    results_csv = os.path.join(save_dir, "results.csv")
    with open(results_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "true_label", "predicted_label", "confidence", "correct"])

        all_preds, all_labels = [], []

        for (img, label), (raw_img, _) in tqdm(zip(loader, raw_loader), total=len(dataset)):
            img, raw_img = img.to(device), raw_img.to(device)
            output = model(img)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = prob.max().item()
            true_label = label.item()
            filename = dataset.samples[len(all_preds)][0].split(os.sep)[-1]

            # Grad-CAM
            cam = gradcam.generate(img, class_idx=pred)
            overlay = overlay_cam_on_image(raw_img, cam)

            # 保存图像
            save_image(raw_img, os.path.join(save_dir, "originals", filename))
            Image.fromarray(overlay).save(os.path.join(save_dir, "overlays", filename))
            cam_np = (cam.squeeze().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(cam_np).save(os.path.join(save_dir, "heatmaps", filename))

            writer.writerow([
                filename,
                class_names[true_label],
                class_names[pred],
                f"{confidence:.4f}",
                "correct" if true_label == pred else "wrong"
            ])

            all_preds.append(pred)
            all_labels.append(true_label)

    acc, precision, recall, f1 = compute_metrics(all_labels, all_preds)
    print(f"Test Results:\nAccuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

if __name__ == "__main__":
    # 修改路径为你的实际测试集路径
    test_dir = "dataset/test"
    model_path = "./save_weights/best_model.pth"
    save_dir = "results"

    test_model(test_dir, model_path=model_path, save_dir=save_dir)
