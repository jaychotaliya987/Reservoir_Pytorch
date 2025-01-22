import torch
from torch import nn
from torch import optim
import numpy 
import matplotlib.pyplot as plt
import sys
import os

# Ensure the correct path to MackeyGlass module
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from Datasets.MackeyGlass2DDataset import MackeyGlass2DDataset
from Datasets.MackeyGlassDataset import MackeyGlassDataset
from Datasets.MackeyGlass import MackeyGlass
from Models import Echostate

from torch.utils.data import DataLoader

print("Imports Done!\n")


Mglass1 = MackeyGlassDataset(10000, 2, 10, 1)

Echostate1 = Echostate.EchoStates(1, 100, num_layers=1, batch_first=True)

Train_test_Split = 0.8
train_size = int(Train_test_Split * len(Mglass1))
test_size = len(Mglass1) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(Mglass1, [train_size, test_size])

print(len(train_dataset))

# Extract data from the dataset for plotting
train_data = [data[0].numpy() for data in train_dataset]
train_data = numpy.concatenate(train_data, axis=0) 
print(train_data.shape)

plt.plot(train_data)
plt.show()