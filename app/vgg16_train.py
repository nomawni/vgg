import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from dataloader import CustomDataloader
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import multiprocessing
import time
from utils import calculate_time
from dotenv import load_dotenv


def main():
    # Prepare device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get environment variables
    annotations = os.getenv("ANNOTATIONS_FILE")
    matched_images = os.getenv("INPUT_DIR")
    # Define the transform,
    # converts the input image to a PyTorch tensor
    # normalizes the image tensor by subtracting the mean
    # and dividing by the standard deviation along each channel
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Define the dataset
    dataset = CustomDataloader(
        csv_or_json_file=annotations, root_dir=matched_images, transform=transform)

    # Define the batch size
    batch_size = 16
    # Calculate the number of unique column values
    column_values = dataset.annotations["class"].nunique()
    # Define the VGG16 model and move it to the device
    model = vgg16(weights=True)
    # model = vgg16(preTrained=True)
    # We replace the last layer of the classifier by a new linear layer (nn.Linear),
    # whose output size matches the number of classes in the dataset
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, column_values)
    model = model.to(device)

    # Define loss function and optimizer, to measure the dissimilaritis of output prob and true prob
    criterion = nn.CrossEntropyLoss()
    # Define a stochastic gradient descent optimizer
    # Parameters of the model that need to be learned
    # learning rate
    # Momentum to avoid local optime and help accelerate convergence.
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    # number of samples to include in each batch
    # specifies whether to shuffle the samples in the dataset before each epoch,
    # to prevent the model from learning the order of the samples
    # number of worker processes to use for data loading
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Training and validation loop
    num_epochs = 10
    train_accuracy_list, val_accuracy_list = [], []

    # Get current time
    start_time = time.time()
    for epoch in range(num_epochs):
        # Training monde
        model.train()
        # correctly predicted samples, and total number of samples
        correct = 0
        total = 0
        for images, labels in train_dataloader:
            # Moves the input images and labels to the device
            images, labels = images.to(device), labels.to(device)
            # Clears the gradients of all model parameters
            optimizer.zero_grad()
            # feeds the input images through the model to obtain the predicted outputs
            outputs = model(images)
            # Computes the loss between the predicted outputs and
            # the true labels using a specified loss function
            loss = criterion(outputs, labels)
            # Backpropagates the gradients through the network, computing the gradients of the loss
            # with respect to all the model's parameters
            loss.backward()
            # Updates the model's parameters using the computed gradients
            # and the chosen optimization algorithm
            optimizer.step()

            # Obtains the predicted labels by finding the class
            # with the highest probability among the model's outputs
            _, predicted = torch.max(outputs.data, 1)
            # Updates the total and correct counters to calculate the accuracy later
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_accuracy_list.append(train_accuracy)

        # Sets the model to evaluation mode,  disables layers like dropout or batch normalization
        model.eval()
        correct = 0
        total = 0
        # Disables gradient calculation ,
        # reducing memory usage and computation time
        with torch.no_grad():
            for images, labels in val_dataloader:
                # Move input images to the device, cpu or gpu
                images, labels = images.to(device), labels.to(device)
                # Forward pass: feeds the input images through the model to obtain the predicted outputs
                outputs = model(images)
                # Obtains the predicted labels by finding the class
                # with the highest probability among the model's outputs.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracy_list.append(val_accuracy)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Get end time
    end_time = time.time()
    # Calculate elapsed time in seconds
    calculate_time(start_time=start_time, end_time=end_time)
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
