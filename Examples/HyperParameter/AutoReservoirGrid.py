import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from reservoirgrid.helpers import chaos_utils  
from reservoirgrid.helpers import utils  
from reservoirgrid.models.Reservoir import Reservoir
from reservoirgrid.helpers import viz

dataset = np.load("reservoirgrid/datasets/Chaotic/Lorenz.npy", allow_pickle=True)
dataset = utils.normalize_data(dataset[15][1])

X_train, X_val, Y_train, Y_val = utils.split(dataset) 

model = Reservoir(input_dim=3, reservoir_dim=1000, output_dim=3)

best_params = model.optimize(
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    metric_fn=chaos_utils.js_divergence, 
    direction="minimize",
    n_trials=500,
    batch_size=25,
)

predictions = model.predict(initial_input=Y_train, steps=len(Y_val))

predictions = predictions.cpu().numpy()  
Y_val = Y_val.cpu().numpy()  

print(chaos_utils.js_divergence(Y_val, predictions))
viz.compare_components([Y_val, predictions], labels=["True", "Predicted"], title="Optimized Reservoir Prediction").show()
viz.compare_plot([Y_val, predictions], labels=["True", "Predicted"], title="Optimized Reservoir Prediction").show()