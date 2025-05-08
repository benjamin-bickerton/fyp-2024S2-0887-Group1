from train_split import train_model
from test_split import test_model
import os

train_dir = "./dataset/train"
val_dir = "./dataset/val"
test_dir = "./dataset/test"

if __name__ == "__main__":
    train_model(train_dir, val_dir)
    test_model(test_dir)
