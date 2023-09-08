import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
from vgg19_models import VGG19
from PIL import Image
# from torch.utils.tensorboard import SummaryWriter
import time
import multiprocessing
from utils import calculate_time
from dotenv import load_dotenv


def train():
    # Prepare the dataset and DataLoader
    train_dir = os.getenv("OUTPUT_TRAIN_DIR")
    val_dir = os.getenv("OUTPUT_VAL_DIR")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16,
                            shuffle=False, num_workers=4)

    # Define the custom VGG19 model and move it to the device
    num_classes = len(train_dataset.classes)
    model = VGG19(num_classes)
    # Newly Added, it ensures that we have the same output classes as the input classes
    model.classifier[6] = torch.nn.Linear(
        model.classifier[6].in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 25
    train_losses, val_losses, val_accuracies = [], [], []

    # Get current time
    start_time = time.time()
    # writer = SummaryWriter('/mi-projects/logs')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Write loss to TensorBoard
            # writer.add_scalar('Loss/train', loss, epoch)

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(running_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")
    # Close TensorBoard
    # writer.close()
    # Get end time
    end_time = time.time()

    calculate_time(start_time=start_time, end_time=end_time)

    # Plot training and validation loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot validation accuracy
    plt.plot(val_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    load_dotenv()
    train()
