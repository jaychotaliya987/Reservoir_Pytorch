import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path  

#* Create a simple dataset
weight = 0.3 
bias = 0.9
RANDOM_SEED = 42

X = torch.arange(0, 1, 0.001, dtype = torch.float32).unsqueeze(dim=1) 
y = weight * X + bias 

#*splitting the data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


#* Function to plot the data
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
    name = "plot.png"
):
    """Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    
    
    IMAGE_PATH = Path("Learning_Pytorch/models/02_Linear_Model_Exe_Images/")
    IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    IMAGE_NAME = name + ".png"
    IMAGE_SAVE_PATH = IMAGE_PATH / IMAGE_NAME
    plt.savefig(IMAGE_SAVE_PATH)



class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def Linear(self, X):
        return self.weight * X + self.bias
    
    def forward(self, X):
        return self.Linear(X)

torch.manual_seed(RANDOM_SEED)
Linear_model = LinearRegression()

print(f"Initial Weights and Biases: {Linear_model.state_dict()}")

#* Training the model
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(Linear_model.parameters(), lr=0.01)

def fit(epochs, model, loss_fn, optimizer, X_train, y_train):
    for epoch in range(epochs):

        train_loss_values = []
        test_loss_values = []
        epoch_count = []
        
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.inference_mode():
            test_pred = Linear_model(X_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float)) 

        if epoch % 20 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


plot_predictions(name = "non_trained.png")

fit = fit(400, Linear_model, loss_fn, optimizer, X_train, y_train)

with torch.inference_mode():
    plot_predictions(predictions=Linear_model(X_test), name = "trained.png")

def SaveModel(model, name):
    MODEL_PATH = Path("Learning_Pytorch/models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = name + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH) 
    torch.save(model.state_dict(), name)

SaveModel(Linear_model, "02_Linear_model_exercise")