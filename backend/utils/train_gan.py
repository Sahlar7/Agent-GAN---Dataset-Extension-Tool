import torch
from torch.utils.data import TensorDataset, DataLoader
from universal_gan import UniversalGAN
import json
import numpy as np
import pandas as pd
import sys
import os
import zipfile
import tempfile
from torchvision import datasets, transforms

def create_dataloader(file_path, batch_size=64, image_size):
    """
    Creates a DataLoader compatible with gan.train().
    Supports zip files containing image folders (e.g., class_name/filename.jpg).
    """

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".zip":
        # Extract to a temporary directory
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Transform for GAN training (normalize to [-1,1])
        transform = transforms.Compose([
            transforms.Resize((64, 64)),           
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = datasets.ImageFolder(root=temp_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    else:
        raise ValueError(f"Unsupported file format for image data: {ext}")


def get_config_from_json(file_path):
    """
    Loads configuration dictionary from a JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Usage: python train_gan.py <dataset_path> <dataset_loader_path> <config_path>")

    dataset_path = sys.argv[1]
    dataset_loader_path = sys.argv[2]
    config_path = sys.argv[3]

    config = get_config_from_json(config_path)
    gan = UniversalGAN(config)

    dataloader = create_dataloader(dataset_path, config. batch_size=config.get("batch_size", 64))

    print(f"🚀 Starting GAN training on {config['modality']} dataset from {dataset_path}")
    gan.train(dataloader)
    print("✅ Training complete.")
    gan.gene
