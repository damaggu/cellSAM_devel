import os
import torch
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import torch.nn.functional as F

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader




# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.annotations[1] = pd.to_numeric(self.annotations[1], errors='coerce').fillna(0).astype(int)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0].split('.')[0] + '.b0.X.npy')
        image = np.load(img_path)
        image = torch.from_numpy(image).float()
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


# Define the neural network architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4), 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x




if __name__ == "__main__":
    # Set the seed
    pl.seed_everything(42)

    # argsparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--img_dir', type=str, default='/data/user-data/rdilip/cellSAM/dataset/val_tuning/neurips')
    parser.add_argument('--val_dir', type=str, default='/data/user-data/rdilip/cellSAM/dataset/hidden/neurips')
    parser.add_argument('--train_csv_file', type=str, default='bloodcellSheet_tuning.csv')
    parser.add_argument('--val_csv_file', type=str, default='bloodcellSheet_hidden.csv')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    IMG_HEIGHT = args.img_size
    IMG_WIDTH = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate

    # Paths to the dataset directories and CSV files
    train_img_dir = args.img_dir
    val_img_dir = args.val_dir
    # IMG_DIR = '/data/user-data/rdilip/cellSAM/dataset/hidden/neurips'
    train_csv_file = args.train_csv_file
    val_csv_file = args.val_csv_file

    # Define transformations
    train_transform = transforms.Compose([
        # transforms.Resize((IMG_HEIGHT, IMG_WIDTH), antialias=True),
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), antialias=True),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the datasets
    train_dataset = CustomImageDataset(csv_file=train_csv_file, img_dir=train_img_dir, transform=train_transform)
    val_dataset = CustomImageDataset(csv_file=val_csv_file, img_dir=val_img_dir, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # Instantiate the model, loss function, and optimizer
    # model = SimpleCNN().to(device)  # Move the model to the GPU

    # Load a pretrained ResNet model and modify it for binary classification
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Modify the final layer for binary classification
    model = model.to(device)  # Move the model to the GPU

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move data to the GPU
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")


        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))  # Use sigmoid to get probabilities, then round to 0 or 1
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        print()

    print("Training completed.")

    # save the model
    torch.save(model.state_dict(), 'bloodcell_classifier.pth')

    print("Model saved.")