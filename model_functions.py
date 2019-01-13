import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Defines the transforms for the training, validation, and testing sets
def create_datasets(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Loads the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)

    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Uses the image datasets and the trainforms to define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 32, shuffle = True)
    validationloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32)

# Defines a new classifier function for the model
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.5):
        super().__init__()
        
        # Allow for the input of an arbitrary number of hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # Adds a dropout probability of 0.5 to model
        self.dropout = nn.Dropout(p = drop_p)
    
    # Defines the forward pass through the network
    def forward(self, x):
        
        # Iterates through each layer in hidden_layers and applies a linear transformation, relu, and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        
        return x

# Defines a function to calculate loss and accuracy for the validation set   
def validation(model, dataloader, criterion, device):
    model.to(device)
    total_loss = 0
    total = 0
    correct = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model.forward(images)
        total_loss += criterion(outputs, labels).item()
        
        ps = torch.exp(outputs)
        total += labels.size(0)
        correct += (labels.data == ps.max(dim = 1)[1]).sum().item()
    
    accuracy = 100 * correct/total
    
    return total_loss, accuracy

def do_deep_learning(model, trainloader, validationloader, epochs, print_every, criterion, optimizer, device = "cpu"):
    epochs = epochs
    print_every = print_every
    steps = 0


    # Converts the model from CPU to GPU if device is cuda
    model.to(device)

    # Training step (updates the weights using gradient descent)

    # Iterates over each epoch
    for e in range(epochs):
        running_loss = 0

        # Puts model back in training mode, so dropout is used
        model.train()

        # Iterates over all the images in each batch
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1

            # Transfers the images and their labels to GPU or CPU (depending on what the device is set as)
            images, labels = images.to(device), labels.to(device)

            # Calculates the gradients and updates the weights
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                # Puts model in evaluation mode so that dropout is not on
                model.eval()

                # Turns off gradients to calculate validation loss and accuracy on the validation set
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validationloader, criterion, device)

                print("Epoch {}/{}...".format(e+1, epochs),
                      'Training Loss: {:.3f}'.format(running_loss/print_every),
                      'Validation Loss: {:.3f}'.format(validation_loss/len(validationloader)),
                      'Validation Accuracy: {:.3f}'.format(accuracy))

                running_loss = 0
                model.train()
                
def create_checkpoint(save_dir, train_datasets, model, optimizer):
    # Maps the classes in the dataset (flower 1, 2, 3...101, 102) to their indices (0, 1, 2...100, 101) so that the output
    # of the neural network (which is an index) can be mapped to the actual classname which is then mapped to the correct
    # flower name in the cat_to_name dictionary
    model.class_to_idx = train_datasets.class_to_idx

    # Creates a dictionary that can be used to resume training of the neural network
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                  'drop_p': 0.5,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'image_datasets': model.class_to_idx,
                  'model': model,
                  'epoch': 11}

    # Saves the checkpoint dictionary into the file checkpoint.pth
    torch.save(checkpoint, save_dir)

# Loads the checkpoint.pth file so that training of the neural network can be resumed
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Loads number of epochs previously run into variable total_epochs
    total_epochs = checkpoint['epoch']
    
    # Creates model, loads in model state_dict, and loads class to idx conversion into model
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['image_datasets']
    
    # Creates optimizer using optim.Adam then loads in the optimizer state_dict
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return total_epochs, model, optimizer