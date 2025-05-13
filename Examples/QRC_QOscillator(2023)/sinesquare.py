import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from reservoirgrid.models import CQOscRes

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
