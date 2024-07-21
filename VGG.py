import scipy.io
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, n_classes=120):
        super(VGG, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * 12 * 12, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)

        return x

def accuracy(predictions, targets):
    pred_classes = torch.argmax(predictions, dim=1)
    true_classes = torch.argmax(targets, dim=1)
    return float(torch.mean((pred_classes == true_classes).float()))

def process_image(image_path, annotation_path, target_size=(100, 100)):
    image = Image.open(image_path).convert('RGB')
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    processed_image = transform(cropped_image)
    return processed_image

class DogDataset(Dataset):
    def __init__(self, file_list, annotation_list, labels, image_dir, annotation_dir, target_size=(100, 100)):
        self.file_list = file_list
        self.annotation_list = annotation_list
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.target_size = target_size
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_file = self.file_list[idx][0][0]
        annotation_file = self.annotation_list[idx][0][0]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        image = process_image(image_path, annotation_path, self.target_size)
        label = self.labels[idx][0] - 1
        return image, label

# Load train list
train_list = scipy.io.loadmat('lists/train_list.mat')
file_list = train_list['file_list']
annotation_list = train_list['annotation_list']
labels = train_list['labels']
print(f"Number of images: {len(file_list)}")

image_dir = 'images/Images'
annotation_dir = 'annotation/Annotation'
train_dataset = DogDataset(file_list, annotation_list, labels, image_dir, annotation_dir)
train_loader = DataLoader(train_dataset, batch_size=600, shuffle=True)

# Load test list
test_list = scipy.io.loadmat('lists/test_list.mat')
file_list = test_list['file_list']
annotation_list = test_list['annotation_list']
labels = test_list['labels']
print(f"Number of images: {len(file_list)}")
test_dataset = DogDataset(file_list, annotation_list, labels, image_dir, annotation_dir)
test_loader = DataLoader(test_dataset, batch_size=715, shuffle=False)

# Initialize VGG model
model = VGG(n_classes=120)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track loss and accuracy
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []
max_steps = 101
eval_freq = 5

# Training loop
for epoch in range(max_steps):
    model.train()
    running_loss = 0.0
    print(f'Epoch {epoch} started')

    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluate on test set
    model.eval()
    if epoch % eval_freq == 0:
        with torch.no_grad():
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            test_accuracy = test_correct / test_total
            test_accuracy_history.append(test_accuracy)
            test_loss_history.append(test_loss / len(test_loader))

            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            train_accuracy = train_correct / train_total
            train_accuracy_history.append(train_accuracy)
            train_loss_history.append(train_loss / len(train_loader))

            print(f'Epoch {epoch}, Train loss: {train_loss / len(train_loader)}, Train accuracy: {train_accuracy}')
            print(f'Epoch {epoch}, Test loss: {test_loss / len(test_loader)}, Test accuracy: {test_accuracy}')

print('Training finished')

# Plot train and test loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()

# Plot train and test accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label='Train Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()

plt.show()
