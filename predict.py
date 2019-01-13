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
from PIL import Image

from model_functions import *
from utility_functions import *

def main():
    # Adds input arguments and sets defaults
    def get_input_args():
        # Creates an argument parser object
        parser = argparse.ArgumentParser()
        
        # Adds arguments to the parser object
        parser.add_argument('pathtoimage', default = '/home/workspace/aipnd-project/flowers/test/28/image_05230.jpg', help = 'path to image')
        parser.add_argument('checkpoint', default = 'checkpoint.pth', help = 'path to checkpoint for model')
        parser.add_argument('--top_k', type = int, default = 1, help = 'Top-k class probabilities to print')
        parser.add_argument('--category_names', default = '/home/workspace/cat_to_name.json', help = 'path to file with mapping of categories to real names')
        parser.add_argument('--gpu', default = 'cpu', action='store_const', const= '', help = 'whether to use gpu or cpu')
        
        
        # Returns the parser object
        return parser.parse_args()
    
    # Assigns the parser arguments to a variable in_args
    in_args = get_input_args()
    
    # Assigns cuda to the variable device if gpu is passed in as an argument
    if in_args.gpu == '':
        device = 'cuda'
    else:
        device = 'cpu'

    # Creates dictionary mapping each flower name to an integer key
    with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Defines the criterion
    criterion = nn.NLLLoss()
    
    # Loads the model and parameters from the checkpoint.pth file
    total_epochs, model, optimizer = load_checkpoint('checkpoint.pth')

    # Processes the image and predicts the name of the image
    np_image = process_image(in_args.pathtoimage)
    probs, classes = predict(in_args.pathtoimage, model, optimizer, device = device, topk = in_args.top_k)
    
    # Creates a list of the actual flower names matching the classes predicted by model
    possible_flowers = []
    for item in classes:
        possible_flowers.append(cat_to_name[str(item)]) 

    # Prints out top k flower species and their respective probabilities
    final_dic = {'flower_species': possible_flowers, 'probabilities': probs}
    for i in range(in_args.top_k):
        print('\nImage Class {}: {}'.format(i + 1, final_dic['flower_species'][i]), '\nProbability: {}'.format(final_dic['probabilities'][i]))
        
if __name__ == '__main__':
    main()