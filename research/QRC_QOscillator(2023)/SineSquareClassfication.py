import numpy as np
import torch
import seaborn as sns
import matplotlib
matplotlib.use("Qt5agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import CQOscRes
from reservoirgrid.datasets import SineSquare

import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("-------------------------")
print("Imports Done!")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')
print("-------------------------")

sample_len = 1000
dataset = SineSquare(sample_len)
data, label = dataset.get_all()
data = data.flatten()
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=20, random_state=42)

print(train_data.shape)
print(test_data.shape)
print(train_label.shape)
print(test_label.shape)


print("Data shape: ", data.shape)
print("Label shape: ", label.shape)

Q_res = CQOscRes(eps_0=500e6 * np.sqrt(1e-3), input_dim=1, h_truncate=8,
        omega = (10 * 2 * np.pi, 9 * 2 * np.pi),
        kappa = (17 * 2 * np.pi, 21 * 2 * np.pi),
        coupling = 700 * 2 * np.pi,
        time = 100e-9, inference = 100,
        output_dim = 1)

density_matrix = Q_res(train_data)

density_matrix = density_matrix.detach().cpu().numpy()

Q_res.train_readout(train_data, train_label)

print(density_matrix.shape)
print(density_matrix)

plt.plot(density_matrix[:-1,:])
plt.savefig("density_matrix.png")
plt.show()
