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

from PIL import Image

# Preprocesses images so they can be fed to neural network
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Loads image into variable im and finds size of image
    im = Image.open(image)
    width, height = im.size
    
    # Reduces shortest side to 256 pixels while maintaining aspect ratio
    if width < height:
        mod_size = (256, 256/width * height)
    elif height < width:
        mod_size = (256/height * width, 256)
    else:
        mod_size = (256, 256)
    
    im.thumbnail(mod_size)
    
    # Recalculates width, height because size of thumbnail is different from size of original image
    width, height = im.size
    
    # Crops out center 224x224 section of image
    left = (width - 224)/2
    right = (width + 224)/2
    upper = (height - 224)/2
    lower = (height + 224)/2 
    
    im = im.crop((left, upper, right, lower))
    
    # Converts image to ndarray
    np_image = np.array(im)
    np_image = np_image/255
    
    transform = transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
    
    # Converts image to tensor and normalizes color channels
    np_image = torch.tensor(np_image)
    np_image = transform(np_image)
    
    # Converts image back to ndarray and makes color channel the first dimension
    np_image = np_image.numpy()
    np_image = np.ndarray.transpose(np_image, (2, 0, 1)) 
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, optimizer, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Preprocesses the image
    processed_image = torch.from_numpy(process_image(image_path))
    processed_image = processed_image.unsqueeze_(0).view(1, 3, 224, 224)
    processed_image = processed_image.to(device).type(torch.cuda.FloatTensor)
   
    model.to(device)
    
    # Passes the image array into the model and converts the outputs to probabilities
    model.eval()
    with torch.no_grad():
        outputs = model.forward(processed_image)
    probs = torch.exp(outputs)
    
    # Finds the top k probabilities and their indices
    probs, indices = probs.data.topk(topk)
    
    probs, indices = probs.to('cpu'), indices.to('cpu')
    
    # Converts the probabilities and indices to ndarrays and then lists
    probs = probs[0].numpy().tolist()
    indices = indices[0].numpy().tolist()
    
    # Reverses the class_to_idx dictionary to create an idx_to_class dictionary
    idx_to_class_list = []
    [idx_to_class_list.append((b, a)) for a, b in model.class_to_idx.items()]
    idx_to_class_dict = dict(idx_to_class_list)
    classes = [idx_to_class_dict[index] for index in indices]
    
    return probs, classes

def plot_solution(image_path, model):
    
    # Creates a figure
    plt.figure(figsize = [6, 10])

    # Creates the 1st subplot
    ax = plt.subplot(2, 1, 1)
    
    # Sets up title of graph
    flower_num = image_path.split('/')[2]
    title = cat_to_name[flower_num]
    
    # Displays image with title and no axes
    image = process_image(image_path)
    imshow(image, ax, title)
    plt.title(title)
    plt.axis('off')
    
    # Calculates probabilities and classes
    probs, classes = predict(image_path, model)

    # Creates a list of the actual flower names matching the classes predicted by model
    possible_flowers = []
    for item in classes:
        possible_flowers.append(cat_to_name[str(item)]) 

    # Creates a pandas dataframe of top 5 flower species and their respective probabilities
    final_dic = {'flower_species': possible_flowers, 'probabilities': probs}
    final_dataframe = pd.DataFrame(data = final_dic)

    # Creates the 2nd subplot
    plt.subplot(2, 1, 2)
    
    # Creates a bar graph that has the probabilities of each flower species as counts
    base_color = sb.color_palette()[0]
    sb.barplot(data = final_dataframe, x = 'probabilities', y = 'flower_species', color = base_color)
    plt.ylabel('Flower Species')
    plt.xlabel('Probabilities (%)')

    # Converts the x-axis labels to percentages
    locs_x, labels_x = plt.xticks()
    plt.xticks(locs, locs*100, rotation = 40);