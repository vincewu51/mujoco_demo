"""
Tiny CNN classifier for MuJoCo synthetic dataset
GPU version (uses CUDA if available)
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MujocoDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)

def train_gpu(dataset_dir="synthetic_dataset", epochs=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    dataset = MujocoDataset(dataset_dir, os.path.join(dataset_dir, "labels.csv"), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss, correct = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}: loss={total_loss:.3f}, acc={acc:.3f}")

    torch.save(model.state_dict(), os.path.join(dataset_dir, "tinycnn_gpu.pth"))


if __name__ == "__main__":
    train_gpu()
