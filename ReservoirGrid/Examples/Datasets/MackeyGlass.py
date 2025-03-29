#This file is property of the ReservoirGrid project. By Jay D. Chotaliya from University of Kassel, Germany.

import numpy
import torch
from scipy.integrate import solve_ivp

class MackeyGlass(torch.utils.data.Dataset):
    """This is a Mackey Glass dataset implementation in Pytorch.
    It will employ the dataset for neural network training and testing."""
    def __init__(self,
                Beta=0.2,
                Gamma=0.1,
                Alpha=0.2,
                N=10, 
                Tau=17,
                T_max=300,
                Dt=0.1,
                History=1.2):
        
        super(MackeyGlass, self).__init__()
        
        self.Beta = Beta
        self.Gamma = Gamma
        self.Alpha = Alpha
        self.N = N 
        #? Number of samples in the dataset, Ex: N=10 implies the data will be divided into 10 segments.
        self.Tau = Tau
        self.T_max = T_max
        self.Dt = Dt
        self.History = History
        self.data = self.generate_data()

    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        return self.data[index]
    
    def generate_data(self):

        data = numpy.zeros(int(self.T_max/self.Dt))
        dxdt = numpy.zeros(int(self.T_max/self.Dt))
        
        for i in range(len(dxdt)):
            if i==0:
                dxdt[i] = self.History
            else:
                dxdt[i] = self.Beta*dxdt[i-1]/(1+dxdt[i-1]**self.Alpha) - self.Gamma*dxdt[i-1]

        data = solve_ivp(lambda t, y: numpy.interp(t - self.Tau, numpy.arange(0, self.T_max, self.Dt), dxdt), 
                         [0, self.T_max], y0=[self.History], method='RK45', 
                         t_eval=numpy.arange(0, self.T_max, self.Dt))

        return data 
