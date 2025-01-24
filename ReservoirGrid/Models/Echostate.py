import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

class EchoStates(nn.RNN):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(EchoStates, self).__init__(input_size, hidden_size, 
                                         num_layers=num_layers, 
                                         batch_first=batch_first, 
                                         nonlinearity='tanh')
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Customize the initialization of the recurrent weights
        self.reset_parameters()
        print("Echo State Module Loaded \n")

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param, a= math.sqrt(5))
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(param)
                bound = 1 / math.sqrt(param.size(0))
                nn.init.uniform_(param, -bound, bound)

    def forward(self, x, h_0=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        output, h_n = super(EchoStates, self).forward(x, h_0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n


