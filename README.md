**Tags:** #Neural_Networks #Reservoir_Computing #Notes #Python #Machine_Learning #Documentation

# General Training Routine for RC

1. **Data Preparation**: For training a Neural Network (NN), you’ll need ready data. Preparing the data requires:
   - **Training Data**
   - **Testing Data**

   After you train the model, you make predictions with unseen data.

2. **Model**: You create a model with *PyTorch*. Your model takes the input data to compute the logic or result of the forward method. Your model will only perform the logic and not make predictions. The model will also initialize the weights and biases and have a forward method.

3. **Training**: The training loop will adjust the weights and biases based on the error computed from the model’s prediction.

---

### Training ESN

Echo State Networks (ESNs) are a special kind of Recurrent Neural Network (RNN). They have LSTM (Long Short-Term Memory) properties, meaning they will have an influence of past activation on the following activation. However, ESNs do not carry out full Back Propagation Through Time (BPTT). Hence, they are easier to train and have hyperparameters that are easier to configure.

1. **Data Preparation**: 
   The data used for training is a stream of input sequences with temporal connections. This means that the input sequence captures the dynamics of a physical system. 

   - **Input sequence shape** must be $(sequence\_length, input\_dim)$.

   For training, you can split the input sequence into 80-20 for training and testing. The longer the model is expected to predict, the higher the errors it will make. With a sufficiently long training sequence, you can train the model and then predict the input sequence beyond the available data.

2. **Model**: 
   The model here is a simple RNN with `grad=false`. This means that, except for $W_{out}$, we only train the output weights. We initialize the weights and biases randomly. The model will use the [[General RC|RNN activation]]. When we feed the sequence of inputs, the model will:
   
   - Initialize the Weights and Biases
   - Create the network with pre-selected nodes
   - Pass the input through the `Forward()` method to obtain the activation

   When the model is passed an input sequence of type `torch.tensor`, it will automatically compute the forward pass:

   ```python
   esn = ESN(input_dim=len(train_data), reservoir_dim=200, output_dim=1)
   esn(train_data)  # this will perform forward pass
   ```

3. **Training**: After the forward pass, we will train the final weights and biases using linear regression. After training, the model will predict the output at $t+1$ using the data at time $t$. The training loop will iterate for $t$ number of epochs to adjust the output weights.
 

---

### Practical ESN Tips

1. **Data Preparation**:
    
    - **Training and Testing Data**: Split your input sequence into **80% for training** and **20% for testing**. This is essential for ensuring that your model generalizes well and avoids overfitting to the training data.
        
    - **Sequence Length**: The model performs better when the training sequence is long enough to capture the dynamics of the system. However, the longer the prediction horizon (how far ahead the model predicts), the larger the errors you may encounter.
        
2. **Model Details**:
    
    - **RNN with `grad=false`**: The recurrent weights are not updated during training, and only the output weights $W_{out}$ are learned using a simple linear regression model.
        
    - **Forward Pass**: During the forward pass, the model takes the input data, computes the activations through the reservoir (using a simple RNN update rule), and outputs the predicted value using a linear readout.
        
3. **Training**:
    
    - **Epochs**: The training loop will iterate through multiple **epochs**. During each epoch, the model will adjust the output weights to minimize the error. The number of epochs can be tuned to optimize the performance.
        
    - **Linear Regression for Output Weights**: After the forward pass, the output weights $W_{out}$ are trained through linear regression. The error is minimized by adjusting these weights so that the output matches the target data as closely as possible.
        

---

### Echo State Network (ESN) Explanation

Echo State Networks are a type of RNN that use a **reservoir** to map inputs to a high-dimensional space. The network consists of an input layer, a reservoir layer, and a readout layer. The reservoir is not trained but instead has a random, fixed weight matrix. The only trained part is the output layer.

#### Mathematical Details:

- **Input Sequence**: The input sequence $\mathbf{u}(n) \in \mathbb{R}^{N_u}$ and the desired target $\mathbf{y}^{target} \in \mathbb{R}^{N_y}$ are used for training. The model aims to replicate the output $\mathbf{y}(n)$ as closely as possible to the target $\mathbf{y}^{target}(n)$, with the least Root Mean Squared Error (RMSE), given by:
$$

E(y, y^{target}) = \frac{1}{N_y} \sum_{i=1}^{N_y} \sqrt{\frac{1}{T} \sum_{n+1}^{T}(y_i - y^{target}_i(n))^2}
$$
- You can normalize the RMSE by dividing it by the variance, resulting in the Normalized RMSE (NRMSE), which should range from 0 to 1.
    
- **Reservoir Dynamics**: The update for the reservoir is given by:
    
$$
\mathbf{\tilde{X}} = \tanh (\mathbf{W}^{in} [1; \mathbf{u}(n)] + \mathbf{W}\mathbf{X}(n - 1))
$$
The state $\mathbf{X}(n)$ is then updated as:
$$
\mathbf{X}(n) = (1 - \alpha) \mathbf{X}(n-1) + \alpha \mathbf{\tilde{X}}(n)
$$
Here, $\mathbf{\tilde{X}} \in \mathbb{R}^{N_X}$ is the update of the reservoir activations $\mathbf{X} \in \mathbb{R}^{N_X}$, and $\mathbf{W} \in \mathbb{R}^{N_X \times N_X}$ is the weight matrix of the reservoir. The input to the reservoir is represented by $\mathbf{W}^{in} \in \mathbb{R}^{N_X \times (1+N_u)}$, and $[.;.]$ represents vertical concatenation.

- The first activation $\mathbf{\tilde{X}(1)}$ is initialized to 0, i.e., $\mathbf{X(0)} = 0$.
    
- **Readout Layer**: The final output is given by:
    
$$
\mathbf{y}(n) = \mathbf{W}^{out} [1; \mathbf{u}(n); \mathbf{X}(n)]
$$
The readout layer combines the input, reservoir activation, and a bias term to compute the output.

#### Key Parameters:

- **Size of Reservoir**: Choose a reservoir large enough to capture the dynamics of the system. A common size is $10^4$. For i.i.d (Independent identically distributed) tasks, the size of the reservoir should match the size of the input data, but for time-dependent tasks, it might be significantly smaller.
    
- **Sparsity**: The sparsity refers to how many elements of the reservoir weight matrix are set to zero. A sparse weight matrix helps with computational efficiency.
    
- **Spectral Radius**: The spectral radius is the maximal absolute eigenvalue of the reservoir weight matrix $\mathbf{W}$. It controls the memory capacity of the network. For the echo state property to hold, $\rho(W) < 1$, but it can work with values significantly higher. A larger spectral radius is generally better for tasks requiring longer memory of the input.
    
- **Leaking Rate $\alpha$**: The leaking rate determines how much of the previous state is retained. It controls the memory properties of the reservoir.
    

---

With these practical tips and explanations, you should be able to build and train your Echo State Network (ESN) efficiently and understand the key parameters and data handling for effective training.

# ReservoirGrid/\_datasets

The library have some built in datasets to utilize and get you started on using RC. The main game then is to fine tune and make your own Reservoir and use it to predict the dataset.
\_datasets include following

```
├── _datasets
│   ├── ADANIPORTS.csv
│   ├── __init__.py
│   ├── LorenzAttractor.py
│   ├── MackeyGlassDataset.py
```

The datasets here is mainly for internal use but users can utilize them if they so please. I used them mainly in the example. I will not maintain them with users in mind.
## ReservoirGrid/\_datasets/MackyGlassDataset.py

 This is the constructor for the Macky-Glass Dataset:

```python
# Constructor
def __init__(self, sample_len, n_samples, tau=17, seed=None):
    self.sample_len = sample_len # sample len in timesteps
    self.n_samples = n_samples   # Number of samples, Number of datasets to be generate. i.e. n_samples = 2 will generate 2 distict datasets 
    self.tau = tau
    self.delta_t = 10
    self.timeseries = 1.2
    self.history_len = tau * self.delta_t
    if seed is not None:
        torch.manual_seed(seed)
```


This dataset will give you the Macky_glass  object with inputs and targets. Sample line to get the dataset would be,
```python
Mglass1 = MackeyGlassDataset(10000, 5, tau=17, seed=0)
inputs, targets = Mglass1[0]
```

The call will generate the object but not the dataset. The second assignment operation will generate the dataset. The argument in the `Mglass1[]` is for the number of sample. This can be useful when you want to have multiple samples, from same object. Also the target is well suited for the reservoir training because the targets are shifted one forward from inputs so you don't have to do it manually every time.
