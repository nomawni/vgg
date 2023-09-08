import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from torch.utils.data import random_split
import matplotlib.pyplot as plt
# Import the custom dataloser
from dataloader import CustomDataloader
# Prepare device for training
from torchvision import transforms
from torch.utils.data import DataLoader
import multiprocessing
import os
from dotenv import load_dotenv
import time
# from torch.utils.tensorboard import SummaryWriter
from utils import calculate_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annotations = os.getenv("ANNOTATIONS_FILE")
    matched_images = os.getenv("INPUT_DIR")
    # Define the transform
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Define the dataset
    dataset = CustomDataloader(
        csv_or_json_file=annotations, root_dir=matched_images, transform=transform)
    # Define the batch size
    # batch_size = 32
    batch_size = 16
    # Calculate the number of unique column values
    column_values = dataset.annotations["class"].nunique()
    # Define the VGG19 model and move it to the device
    model = vgg19(weights=True)
    # model = vgg19(pretrained=False)
    num_features = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_features, len(column_values))
    model.classifier[6] = nn.Linear(num_features, column_values)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training and validation loop
    num_epochs = 10
    train_accuracy_list, val_accuracy_list = [], []

    # Calculate the time is took to train the model
    start_time = time.time()
    # writer = SummaryWriter('/mi-project/logs')
    for epoch in range(num_epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Write loss to TensorBoard
            # writer.add_scalar('Loss/train', loss, epoch)
        train_accuracy = correct / total
        train_accuracy_list.append(train_accuracy)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracy_list.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    end_time = time.time()

    # Calculate elapsed time
    calculate_time(start_time=start_time, end_time=end_time)
    # writer.close()
    # Plot the training and validation accuracy
    plt.plot(train_accuracy_list, label='Training Accuracy')
    plt.plot(val_accuracy_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    load_dotenv()
    main()
