# check_labels.py
import os
from torchvision import datasets

dataset_root = '/home/hlcv_team010/thermal-uda-attention/dataset_dir/uda_data/flir/train'
dataset = datasets.ImageFolder(root=dataset_root)

print("Class-to-index mapping:", dataset.class_to_idx)
labels = [label for _, label in dataset.samples]
print("Labels in dataset:", set(labels))
