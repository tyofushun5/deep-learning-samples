import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 128,
    "learning_rate": 1e-4,
    "weight_decay": 5e-4,
    "epochs": 30
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
validation_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
names = ("plane", "cat", "bird", "deer", "dog", "frog", "horse", "ship", "truck")

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=config["batch_size"], shuffle=False)


model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(config["device"])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

losses = []
accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(config["epochs"]):
    running_loss = 0.0
    running_accuracies = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(config["device"])
        labels = labels.to(config["device"])
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        prediction = torch.argmax(outputs, dim=1)

    running_loss /= len(train_loader)
    running_accuracies /= len(train_loader)
    losses.append(running_loss)
    accuracies.append(running_accuracies)
    print("epoch : ", epoch, "loss : ", running_loss, "accuracy : ", running_accuracies)

    validation_loss = 0.0
    validation_accuracies = 0.0
    for validation_imgs, validation_labels in validation_loader:
        validation_imgs = validation_imgs.to(config["device"])
        validation_labels = validation_labels.to(config["device"])
        validation_outputs = model(validation_imgs)
        validation_loss = criterion(validation_outputs, validation_labels)
        validation_loss += validation_loss.item()
        validation_prediction = torch.argmax(validation_outputs, dim=1)
        validation_accuracies += torch.mean(validation_prediction.eq(validation_labels).float())

    validation_loss /= len(validation_loader)
    validation_accuracies /= len(validation_loader)
    val_losses.append(validation_loss)
    val_accuracies.append(validation_accuracies)
    print("validation loss : ", validation_loss, "validation accuracy : ", validation_accuracies)