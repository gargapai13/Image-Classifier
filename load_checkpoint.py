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

# Sets up a testing directory
test_dir = '/home/workspace/aipnd-project/flowers/test/'

# Defines the transforms for the testing set
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Loads the dataset with ImageFolder
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

# Uses the test image dataset and the trainforms to define the dataloader
testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32)
 
# Defines the criterion and to use gpu
criterion = nn.NLLLoss()
device = 'cuda'

# Load the model and parameters from the checkpoint.pth file
total_epochs, model, optimizer = load_checkpoint('checkpoint.pth')

# Runs the model on the testing dataset
model.eval()
test_loss, accuracy = validation(model, testloader, criterion, device)
print('Testing Loss: {:.3f}'.format(test_loss),
      'Accuracy: {:.3f}'.format(accuracy))