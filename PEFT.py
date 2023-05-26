# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:28:34 2023

@author: tye
"""
#%% Setup
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor
import os
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

rel_path = os.path.realpath(__file__)

#%% Preprocessing Methods
def one_hot_encode(data, dictionary):
    nrow = len(data)
    ncol = len(dictionary)
    
    
    matrix = np.zeros(shape = (nrow,ncol))
    for i in range(nrow):
        for k in range(len(data[i])):
            col = dictionary[data[i][k]]
            matrix[i,col] = 1.0
    return matrix







#%%
batch_size = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
