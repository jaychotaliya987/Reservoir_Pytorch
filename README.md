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

# Echo State Network (ESN)

## The Model

ESNs are supervised learning ML task. The input sequence $\mathbf{u}(n) \in \mathbb{R}^{N_{u}}$ and a desired target (True data) is $\mathbf{y}^{target} \in \mathbb{R}^{N_{y}}$. The learning task is to replicate the $\mathbf{y}(n)$, the model output, with $\mathbf{y}^{target}(n)$ with least Root-Mean-Square-error (RMSE), given by $\mathbf{E}$,
$$
E(y, y^{target}) = \frac{1}{N_{y}} \sum_{i=1}^{N_{y}} \sqrt{ \frac{1}{T} \sum_{n+1}^{T}(y_{i}-y_{i}^{target}(n))^2}
$$
We can also normalize the RMSE, by dividing it with variance. This results in NRMSE. NRMSE does not scale with arbitrary scaling of the target.  **NRMSE should achieve the accuracy of zero to one.**

ESNs typically use RNN type, leaky-integrated discrete-time continuous-value units. Update equation is as follow:

$$
\mathbf{\tilde{X}} = \tanh (\mathbf{W}^{in} [1; \mathbf{u}(n)] + \mathbf{W}\mathbf{X}(n − 1))
$$

$$
\mathbf{X}(n) = (1- \alpha) \mathbf{X}(n-1) + \alpha \mathbf{\tilde{X}}(n)
$$

Here the $\mathbf{\tilde{X}} \in \mathbb{R}^{N_{X}}$ is the update of the reservoir neuron activations $\mathbf{X} \in \mathbb{R}^{N_{X}}$.  The reservoir weights $\mathbf{W}\in \mathbb{R}^{N_{X} \times N_{X}}$ and input to reservoir matrix $\mathbf{W^{in}} \in \mathbb{R}^{N_{X} \times (1+N_{u})}$.  operation $[ . ;  . ]$  is a vertical vector concatenation. ==For the first activation $\mathbf{\tilde{X}(1)}$, $\mathbf{X(0)}$ is initialized to be zero.== 

The linear readout is,
$$
\mathbf{y}(n) = \mathbf{W}^{out} [1; \mathbf{u}(n);\mathbf{X}(n)]
$$

# Reservoir Production

The reservoir serves 2 purposes, **(i)** Nonlinear expansion for the input $\mathbf{u}(n)$ and, **(ii)** As a memory of the inputs $\mathbf{u}(n)$. 

> $\mathbf{u}(n)$ is a input. $\mathbf{X}$ is a reservoir activation or weighted input that goes through activation function. $\mathbf{\tilde{X}}$ is a reservoir activation update. The update equation is quite misleading as it is not used to update anything it generates a new activation for the input with history of the past inputs through $\mathbf{\tilde{X}(n-1)}$.

A RC can be seen as, a nonlinear high-dimensional expansion $\mathbf{X}(n)$ of the input signal $\mathbf{u}(n)$. For classification tasks, input data $\mathbf{u}(n)$ which are not linearly separable in the original space $\mathbb{R}^{N_{u}}$ , often become so in the expanded space $\mathbb{R}^{N_{x}}$ of $\mathbf{X}(n)$, where they are separated by $\mathbf{W}^{out}$ .
## Hyper-parameters

The global parameters of the Reservoir are,

- **Size of the Reservoir :** Choose the reservoir to be as big as computationally possible. Common tasks will leave the reservoir of the size, $10^4$. Optimize the parameters with smaller reservoir and then scale it to bigger reservoir. For i.i.d (Independent identically distributed) tasks, the size should be of $N_{u}$ if the input is of size $u$. But for time dependent tasks it will compress significantly. 

- **Sparsity :** Sparsity is a low priority parameter. It refers to the elements in $W^{in}$ equals to zero. It helps with the computational speed ups of the reservoir. This is called **fan-out** number, The number of the neurons connected to other neurons.

- **Spectral Radius :**  Spectral radius is the central parameter of the ESN. It is the maximal absolute eigenvalue of the weight matrix $W$. a random sparse $W$ is generated; its spectral radius $ρ(W)$ is computed; then $W$ is divided by $ρ(W)$ to yield a matrix with a unit spectral radius. Then that can be tuned with the tuning procedure.
	- The echo state property of the reservoir which is: The state of the reservoir $X(n)$ is uniquely defined by the fading history of the input $u(n)$. 
	- Echo state property holds for $\rho(W) < 1$ but practically it can work even with significantly high values of $\rho(W)$. Practically $\rho(W)$ should be selected based on the performance of the network.
	-  **The spectral Radius should be greater in tasks requiring longer memory of the input.**

- **Input scaling:** Input scaling `a` is a range of interval [-1;1] from which the $W^{in}$ is sampled. 
	- Generally, whole $W^{in}$ is scaled with single value of input scaling. But for performance it is recommended to use the different scaling for the biases.
	- For normally distributed $W^{in}$ use the standard deviation as a scaling parameter
	- **Always normalize the input signal.** Normalizing avoids the outliers that can throw the reservoir out of usual trajectory of the input.
	- for **linear tasks,** $W^{in}$ should be small(close to 0). Where the tanh() activation is somewhat linear as well. while bigger $W^{in}$(bigger `a`) will have neurons quickly saturating to -1 or 1. 
	- scaling of $W^{in}$ along with the scaling of $W$ determines the effect of current input and previous state on current state from the equation. $$ \mathbf{\tilde{X}} = \tanh (\mathbf{W}^{in} [1; \mathbf{u}(n)] + \mathbf{W}\mathbf{X}(n − 1))
		 $$
	- Different principle component of $\mathbf{u}(n)$ in $x(n)$ is roughly proportional to square root of their magnitude in $\mathbf{u}(n)$. (Michiel Hermans and Benjamin Schrauwen. Memory in reservoirs for high dimensional input. In Proceedings of the IEEE International Joint Conference on Neural Networks, 2010 (IJCNN 2010), pages 1–7, 2010), Then it might be helpful to remove component that holds no useful information.
	
- **Leaky Rate:** It is speed of reservoir update dynamics discretized in time. The paper mentions that it is empirically similar to resampling from $\mathbf{u}(n)$. but it is unclear to me how and why. General guide line is to match the speed of the dynamics. which again is unclear, The practical approach is to set the leaky rate high for fast changing signal, and low for steady and periodic signals.
	- There is a better way to do this, **Autocorrelation:** Set the $\alpha = \frac{1}{\tau}$ where $\tau$ is a time step when the difference in the intensity between two point is down 37%
	- look for the paper for more details :  Mantas Lukoševičius, Dan Popovici, Herbert Jaeger, and Udo Siewert. Time warping invariant echo state networks. Technical Report No. 2, Jacobs University Bremen, May 2006.

## Practical Approach to Reservoir Production


With these practical tips and explanations, you should be able to build and train your Echo State Network (ESN) efficiently and understand the key parameters and data handling for effective training.

First point in designing a resrvoir is that jumping through every parameter will quickly escilate and one will lose track of what parameter does what it is benificial to fix some parameters and identify parameters that matters the most.

For ESNs there are three main parameter:
1. Input Scaling
2. Spectral Radius
3. Leaky Rate

reservoir size is generally an external restriction. For most of chaotic serieses (ex. Lorenz) size of 1500 serves well, but after that the returns are diminishing.

Another practical guide is to plot the reservoir activation to see if the reservoir is capturing dynamics of the input signal, Ideally the reservoir states should have lypunov comparable to the input time series. This will help you adjust the spectral radius if the reservoir states are falling into a low component modes(say stuck in one attractor in lorenz) after certain time, if they do then the spectral radii is low. or if it is lost in the noise, spectral radius is big, for effective capture you need to dial it back a little. 

The general guide about the range is that performance improvement is not found in the nerrow parameter range. Reservoir does not need fine tuning. A general range of parameter will result in the similar performance. 


___

# ReservoirGrid/datasets

The library have some built in datasets to utilize and get you started on using RC. The main game then is to fine tune and make your own Reservoir and use it to predict the dataset.
datasets include following

```
├── _datasets
│   ├── ADANIPORTS.csv
│   ├── __init__.py
│   ├── LorenzAttractor.py
│   ├── MackeyGlassDataset.py
```

The datasets here is mainly for internal use but users can utilize them if they so please. I used them mainly in the example. I will not maintain them with users in mind.
## ReservoirGrid/datasets/MackyGlassDataset.py

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
