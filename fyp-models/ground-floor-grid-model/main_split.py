from train_split import train_model
from test_split import test_model
import os

# base_dir = os.path.dirname(os.path.abspath(__file__))
# train_dir = os.path.join(base_dir, 'dataset', 'train')
# val_dir = os.path.join(base_dir, 'dataset', 'val')
# test_dir = os.path.join(base_dir, 'dataset', 'test')

train_dir = "./dataset/train"
val_dir = "./dataset/val"
test_dir = "./dataset/test"

if __name__ == "__main__":
    train_model(train_dir, val_dir)
    test_model(test_dir)
