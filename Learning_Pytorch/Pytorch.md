# PyTorch Cheat Sheet

## 1. Initialization

### Importing PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
```

### Creating Tensors

```python
# Creating a tensor
x = torch.tensor([1, 2, 3])
y = torch.Tensor([4, 5, 6])

# Creating tensors with specific shapes
a = torch.zeros(2, 3)  # 2x3 matrix of zeros
b = torch.ones(2, 3)   # 2x3 matrix of ones
c = torch.eye(3)       # 3x3 identity matrix
d = torch.rand(2, 3)   # 2x3 matrix with random values
```

## 2. Tensor Operations

### Basic Operations
```python
# Addition
z = x + y
z = torch.add(x, y)

# Subtraction
z = x - y
z = torch.sub(x, y)

# Multiplication
z = x * y
z = torch.mul(x, y)

# Division
z = x / y
z = torch.div(x, y)
```

### Matrix Multiplication
```python
a = torch.rand(2, 3)
b = torch.rand(3, 2)
c = torch.mm(a, b)  # Matrix multiplication
```

### Aggregation Operations
```python
# Sum
sum_x = torch.sum(x)

# Mean
mean_x = torch.mean(x.float())

# Maximum and Minimum
max_x = torch.max(x)
min_x = torch.min(x)
```

### Reshaping Tensors
```python
x = torch.rand(2, 3)
y = x.view(3, 2)     # Reshape without changing data
z = x.reshape(3, 2)  # Reshape
```

## 3. Autograd: Automatic Differentiation

### Enabling Gradients
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
```

### Backward Pass
```python
y = x + 2
z = y * y * 2
z = z.mean()

# Compute gradients
z.backward()

# Gradients
print(x.grad)
```

## 4. Neural Network Components

### Defining a Neural Network
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
```

### Loss Function
```python
criterion = nn.MSELoss()
```

### Optimizer
```python
optimizer = optim.SGD(net.parameters(), lr=0.01)
```

## 5. Training a Neural Network

### Training Loop
```python
for epoch in range(100):
    optimizer.zero_grad()   # Zero the gradients
    outputs = net(inputs)   # Forward pass
    loss = criterion(outputs, targets)  # Compute loss
    loss.backward()         # Backward pass
    optimizer.step()        # Update weights
```

## 6. CUDA: Using GPU

### Check GPU Availability
```python
torch.cuda.is_available()
```

### Moving Tensors to GPU
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
```

### Moving Model to GPU
```python
net.to(device)
```

## 7. Saving and Loading Models

### Save Model
```python
torch.save(net.state_dict(), 'model.pth')
```

### Load Model
```python
net = Net()
net.load_state_dict(torch.load('model.pth'))
net.eval()
```

## 8. Loading Data

### Using DataLoader
```python
from torch.utils.data import DataLoader, Dataset

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self):
        # Initialize data, download, etc.
        pass

    def __len__(self):
        # Return the number of samples
        return 100

    def __getitem__(self, idx):
        # Get a sample
        return torch.tensor([1.0]), torch.tensor([0.0])

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```



# SUMMARY 

## Tensor Operations

### Creation
- `torch.tensor(data)`: Create a tensor from data.
- `torch.zeros(size)`: Create a tensor of zeros with given size.
- `torch.ones(size)`: Create a tensor of ones with given size.
- `torch.eye(n)`: Create an identity matrix of size n x n.
- `torch.arange(start, end, step)`: Create a 1-D tensor with values from start to end-1 with given step.
- `torch.linspace(start, end, steps)`: Create a 1-D tensor with steps equally spaced values between start and end.

### Indexing and Slicing
- `tensor[index]`: Get value at index.
- `tensor[start:end]`: Slice tensor from start to end-1.
- `tensor[:, index]`: Get value at index along a specific dimension.
- `tensor[..., start:end]`: Slice tensor along multiple dimensions.

### Reshaping
- `tensor.view(shape)`: Reshape tensor to given shape.
- `tensor.reshape(shape)`: Reshape tensor to given shape.
- `tensor.squeeze(dim)`: Remove single-dimensional entries from the shape of a tensor.
- `tensor.unsqueeze(dim)`: Add a dimension of size one at the specified position.

### Concatenation
- `torch.cat(tensors, dim)`: Concatenate tensors along a specific dimension.
- `torch.stack(tensors, dim)`: Stack tensors along a new dimension.

## Mathematical Operations

### Basic Operations
- `torch.add(tensor1, tensor2)`: Element-wise addition.
- `torch.sub(tensor1, tensor2)`: Element-wise subtraction.
- `torch.mul(tensor1, tensor2)`: Element-wise multiplication.
- `torch.div(tensor1, tensor2)`: Element-wise division.
- `torch.mm(tensor1, tensor2)`: Matrix multiplication.

### Reduction Operations
- `torch.sum(tensor)`: Sum of all elements in tensor.
- `torch.mean(tensor)`: Mean of all elements in tensor.
- `torch.max(tensor)`: Maximum value in tensor.
- `torch.min(tensor)`: Minimum value in tensor.

## Autograd

### Automatic Differentiation
- `tensor.requires_grad_(True)`: Enable gradient tracking for tensor.
- `tensor.backward()`: Compute gradients of tensor.
- `torch.no_grad()`: Context manager to disable gradient calculation.

## Neural Network Components

### Layers
- `torch.nn.Linear(in_features, out_features)`: Fully connected layer.
- `torch.nn.Conv2d(in_channels, out_channels, kernel_size)`: 2D convolutional layer.
- `torch.nn.ReLU()`: Rectified Linear Unit activation function.
- `torch.nn.Sigmoid()`: Sigmoid activation function.
- `torch.nn.CrossEntropyLoss()`: Cross-entropy loss function.
- `torch.nn.MSELoss()`: Mean squared error loss function.

### Activation Functions
- `torch.nn.ReLU()`: Rectified Linear Unit activation function.
- `torch.nn.Sigmoid()`: Sigmoid activation function.
- `torch.nn.Tanh()`: Hyperbolic tangent activation function.
- `torch.nn.Softmax(dim)`: Softmax activation function.

### Optimizers
- `torch.optim.SGD(params, lr)`: Stochastic Gradient Descent optimizer.
- `torch.optim.Adam(params, lr)`: Adam optimizer.

## Utilities

### Device Management
- `torch.device('cuda:0')`: Device object for CUDA GPU.
- `tensor.to(device)`: Move tensor to specified device.

### Random Number Generation
- `torch.rand(size)`: Create a tensor of random numbers with uniform distribution.
- `torch.randn(size)`: Create a tensor of random numbers with normal distribution.
- `torch.randint(low, high, size)`: Create a tensor of random integers between low (inclusive) and high (exclusive).

### Saving and Loading Models
- `torch.save(model.state_dict(), filepath)`: Save model parameters to file.
- `model.load_state_dict(torch.load(filepath))`: Load model parameters from file.

