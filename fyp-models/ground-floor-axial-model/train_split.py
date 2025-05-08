import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import EfficientNet_B0_Weights
import os
from tqdm import tqdm

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

def plot_confusion_matrix(y_true, y_pred, class_names):
    matrix = torch.zeros(2, 2, dtype=torch.int32)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix.numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.show()

def train_model(train_dir, val_dir, model_path="./save_weights/best_model.pth", epochs=25, batch_size=32, patience=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
        transforms.Normalize([0.61, 0.66, 0.62],
                             [0.44, 0.43, 0.44])
    ])

    train_set = datasets.ImageFolder(train_dir, transform=transform)
    val_set = datasets.ImageFolder(val_dir, transform=transform)
    class_names = train_set.classes

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_val_acc = 0
    best_f1 = 0
    stop_counter = 0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_acc, precision, recall, f1 = compute_metrics(val_targets, val_preds)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print("Early stopping.")
                break

    plot_confusion_matrix(val_targets, val_preds, class_names)
