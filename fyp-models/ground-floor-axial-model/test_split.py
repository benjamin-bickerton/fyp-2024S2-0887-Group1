import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.show()

def test_model(test_dir, model_path="./save_weights/best_axial_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.61, 0.66, 0.62],
                             [0.44, 0.43, 0.44])
    ])

    test_set = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    class_names = test_set.classes

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    acc, precision, recall, f1 = compute_metrics(labels_all, preds_all)
    print(f"Test Results:\nAccuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
    plot_confusion_matrix(labels_all, preds_all, class_names)
