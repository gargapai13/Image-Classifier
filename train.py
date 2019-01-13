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

import argparse
from time import time, sleep
import os

import json

from model_functions import *
from utility_functions import *

def main():
    # Adds input arguments and sets defaults
    def get_input_args():
        # Creates an argument parser object
        parser = argparse.ArgumentParser()
        
        # Adds arguments to the parser object
        parser.add_argument('data_dir', help = 'path to the data directory')
        parser.add_argument('--arch', type = str, default = 'vgg19', help = 'choose a type of vgg architecture for model')
        parser.add_argument('--save_dir', type = str, default = '', help = 'path to use to save the checkpoint file')
        parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'learning rate for the model')
        parser.add_argument('--hidden_units', nargs = '+' , type = int, default = [2048, 1000], help = 'provide a list of size of hidden layers in model with no commas')
        parser.add_argument('--epochs', type = int, default = 11, help = 'number of epochs to run model on')
        parser.add_argument('--gpu', default = 'cpu', action='store_const', const= '', help = 'use gpu or cpu to train model')
        
        # Returns the parser object
        return parser.parse_args()
    
    # Assigns the parser arguments to a variable in_args
    in_args = get_input_args()

    # Sets up a training, validation, and testing directory
    train_dir = in_args.data_dir + 'train/'
    valid_dir = in_args.data_dir + 'valid/'
    test_dir = in_args.data_dir + 'test/'
    
    # Defines the transforms for the training, validation, and testing sets
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
    
    # Sets up a pretrained neural network specified by the arch input
    arch = in_args.arch
    
    if arch == 'resnet18':
        model = models.resnet18(pretrained = True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained = True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained = True)
    elif arch == 'resnet101':
        model = models.resnet101(pretrained = True)
    elif arch == 'resnet152':
        model = models.resnet152(pretrained = True)
    elif arch == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained = True)
    elif arch == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    elif arch == 'inception_v3':
        model = models.inception_v3(pretrained = True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained = True)
    elif arch == 'densenet169':
        model = models.densenet169(pretrained = True)
    elif arch == 'densenet201':
        model = models.densenet201(pretrained = True)
    elif arch == 'vgg11':
        model = models.vgg11(pretrained = True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif arch == 'vgg11_bn':
        model = models.vgg11_bn(pretrained = True)
    elif arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained = True)
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained = True)
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained = True)
   
    # Freezes all parameters in the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Determines the input size to feed the classifier
    input_size = model.classifier[0].in_features

    # Changes the classifier defined in the model
    model.classifier = Network(input_size, 102, in_args.hidden_units, drop_p = 0.5)
    
    # Implements a criterion, optimizer, and gpu if available
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = in_args.learning_rate)
    if in_args.gpu == '':
        device = 'cuda'
    else:
        device = 'cpu'

    '''total_epochs, model, optimizer = load_checkpoint('checkpoint.pth')'''
    
    # Carries out the training and validation steps
    do_deep_learning(model, trainloader, validationloader, in_args.epochs, 40, criterion, optimizer, device)
    
    # Runs the model on the testing dataset
    model.eval()
    test_loss, accuracy = validation(model, testloader, criterion, device)
    print('Testing Loss: {:.3f}'.format(test_loss),
          'Accuracy: {:.3f}'.format(accuracy))
    
    # Saves the network to a specified directory
    if in_args.save_dir:
        create_checkpoint(in_args.save_dir, train_datasets, model, optimizer)
  
if __name__ == '__main__':
    main()