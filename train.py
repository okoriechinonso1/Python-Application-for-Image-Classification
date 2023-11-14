# Imports here
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.models as models
import signal
from contextlib import contextmanager
import requests
import json
#from torchvision.models import VGG16_Weights
from torch import nn
from torch import optim
#from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms as transforms
import os
import shutil
import argparse


# Define a function to parse input arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('data_dir', type=str, default='flowers', nargs='?', help='Data directory for training images')     
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()

# Define a main function to handle the training process
def main():
    args = get_input_args()
    
    
    
    def load_data(filepath, means, standard_deviation):
        # Load and preprocess the dataset
        imagenet_data = torchvision.datasets.ImageFolder(filepath)
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True)
    
        train_dir = filepath + '/train'
        valid_dir = filepath + '/valid'
        test_dir = filepath + '/test'
    
        # TODO: Define your transforms for the training, validation, and testing sets
        data_transforms = {
            'train': transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(means, standard_deviation)
            ]),
            'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, standard_deviation)
            ]),
            'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, standard_deviation)
            ])
            }

        # Define directories
        data_dirs = {
            'train': train_dir,
            'valid': valid_dir,
            'test': test_dir
        }


        # TODO: Load the datasets with ImageFolder
        image_datasets = {x: datasets.ImageFolder(data_dirs[x], transform=data_transforms[x])
                          for x in ['train', 'valid', 'test']}

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=32 if x == 'train' else 16, shuffle=True if x == 'train' else         False)
                      for x in ['train', 'valid', 'test']}
        # Combine image_datasets and dataloaders into one dictionary
        data_dict = {
            'image_datasets': image_datasets,
            'dataloaders': dataloaders
        }
        return data_dict

    # Load the model data using the load_model function
    means = [0.485, 0.456, 0.406]
    standard_deviation = [0.229, 0.224, 0.225]
    filepath = 'flowers'
    data = load_data(filepath, means, standard_deviation)

    # TODO: Build and train your network
    torch.cuda.empty_cache()

    # Define your model
    def define_model(arch, num_features, num_classes, device):

        # Initialize the model variable to None
        model = None

        # Load the corresponding pre-trained model based on the architecture
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[0].in_features
        elif arch == 'alexnet':
            model = models.alexnet(pretrained=True)
            num_features = model.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Freeze the parameters of the feature extractor
        for param in model.features.parameters():
            param.requires_grad = False

        # Replace the classifier with a custom one for flower species classification
        # Assume num_classes is the number of flower categories
        classifier = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
            )

        model.classifier = classifier
       
        return model

    num_of_classes = 102
    num_features = 4096
    arch = 'vgg16'
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_model(args.arch, num_features, num_of_classes, device)
    model = model.to(device)
    
    
    # Train the model
    def train_model(model, criterion, optimizer, dataloaders, num_epochs):
 
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                batch_index = 0

                for inputs, labels in dataloaders[phase]:
                    print(f"Processing batch {batch_index + 1} of {phase}")
                    batch_index += 1

                    # Move inputs and labels to the same device as the model
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Debug prints to ensure the tensors are on the correct device
                    optimizer.zero_grad()
                
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward and optimize in train phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            #torch.cuda.empty_cache()  # Clear unused memory

                    # Calculate statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # Clear unused memory after each epoch
            #if torch.cuda.is_available():
                #torch.cuda.empty_cache()
        return model
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    num_epochs = 20
    # Dataloaders is a dictionary containing 'train' and 'valid' DataLoader objects
    dataloaders = data['dataloaders']
    # Move the model to CPU and save the checkpoint
    #model.to('cpu')
    model_trained = train_model(model, criterion, optimizer, dataloaders, num_epochs)
        
    # Save the trained model

    torch.save(model_trained.state_dict(), 'flower_species_classifier.pth')
    
    # Save the checkpoint
    def save_checkpoint(model, architecture, image_datasets, optimizer, epochs, checkpoint_path):
        """
        Saves a checkpoint of the model state.
        Parameters:
            model (torch.nn.Module): The trained model.
            architecture (str): The architecture of the model.
            image_datasets (Dataset): The dataset used for training, with class_to_idx attribute.
            optimizer (torch.optim.Optimizer): The optimizer used during training.
            epochs (int): The number of epochs the model was trained for.
            checkpoint_path (str): Path to save the checkpoint.
        """

        # Attach the class to index mapping to the model
        model.class_to_idx = image_datasets['train'].class_to_idx

        # Create a checkpoint dictionary
        checkpoint = {
            'architecture': architecture,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict(),
            'classifier': model.classifier,  # Save the classifier separately
            'optimizer_state': optimizer.state_dict(),
            'epochs': epochs
            }

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)


    # Assuming model, image_datasets, optimizer, and number of epochs are already defined
    checkpoint_path ='model_checkpoint.pth'
    image_datasets = data['image_datasets']
    epochs = 20
    architecture = args.arch
    save_checkpoint(model, architecture, image_datasets, optimizer, epochs, checkpoint_path)

# Call to main function to run the program
if __name__ == "__main__":
    main()






