# Import necessary libraries
# Other imports for processing image and loading model
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
#from torchvision.models import vgg16, VGG16_Weights


def main(args):
    # Implement code to predict the class from an image file
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

        if checkpoint['architecture'] == 'vgg16':
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            model.classifier = checkpoint['classifier']
        else:
            # Handle other architectures
            pass

        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
        model.to('cpu')  # Move the model to the CPU
        model.device = torch.device('cpu')  # Set the device attribute

        return model

    # Load your model (replace 'your_checkpoint_path.pth' with your checkpoint file path)
    model = load_checkpoint('model_checkpoint.pth')
    checkpoint = torch.load('model_checkpoint.pth', map_location=torch.device('cpu'))
    
    
    def process_image(image_path, means, standard_dev):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns a NumPy array
        '''
        # Open the image
        pil_image = Image.open(image_path)

        # Convert image to RGB if it's not
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Define transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=standard_dev)
            ])

        # Apply transformations to the image
        tensor_image = preprocess(pil_image)

        # Convert to NumPy array
        np_image = tensor_image.cpu().detach().numpy()

        # PyTorch models expect the color channel to be in the first dimension
        # but numpy arrays have it in the third dimension
        # This line is already correct, as the tensor is expected to be in CxHxW format
        np_image = np_image.transpose((1, 2, 0))

        # Correcting the color channel position for display
        np_image = np_image.transpose((2, 0, 1))
        return np_image

    # Example usage
    image_path = "ImageClassifier/assets/Flowers.png"
    means = [0.485, 0.456, 0.406]
    standard_dev = [0.229, 0.224, 0.225]
    image = process_image(image_path, means, standard_dev)


    #Class Prediction
    # Define the function to predict the class of an image
    def predict(image_path, model, means, standard_dev, topk, device='cpu'):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Process the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=standard_dev)
        ])
    
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
    
        # Ensure the model is in evaluation mode
        model.eval()
    
        # Make predictions
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_idxs = probabilities.topk(topk)
    
        # Convert to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_idxs.cpu().numpy()[0]]
    
        return top_probs.cpu().numpy(), top_classes
    
    
    # Define the model architecture and load the checkpoint
    def load_checkpoint(filepath, device='cpu'):
        checkpoint = torch.load(filepath, map_location=device)
    
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        model.to(device)
    
        return model

    # Assuming 'model_checkpoint.pth' is the path to the checkpoint file and 'path_to_image' is the image file
    means = [0.485, 0.456, 0.406]
    standard_dev = [0.229, 0.224, 0.225]
    topk = 5
    model = load_checkpoint('model_checkpoint.pth', device='cpu')
    image_path = "/ImageClassifier/assets/Flowers.png"
    probs, classes = predict(image_path, model, topk=5, device='cpu')
    print(probs)
    print(classes)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('input', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    main(args)
