# Study of Input Discretization Effects on Learning with Reservoir Computing

### Key Points

* The **Dysts dataset** includes an input discretization mechanism, typically tied to a dominant Fourier component or the Lyapunov exponent. This works well for **pseudo-chaotic** or **periodic systems**, where periods are well defined. However, in systems lacking a clear periodic structure, such resampling can **blur important dynamics** or introduce artifacts.

* I’ve implemented a function in `reservoirgrid.helpers.utils` to discretize **chaotic systems** by integrating them at a **fixed time step `dt`**. This approach avoids interpolating onto assumed periodic grids and respects the continuous chaotic dynamics more faithfully.

* There are key functions in `utils` for discretization:

  * `discretization`: Returns a NumPy array structured as `[[points_per_period], [data]]`, with the `data` aligned to the system's internal period estimates. Best suited for periodic or near-periodic systems.
  * `discretization_with_dt`: Returns just the `data` array, sampled at fixed intervals determined by the model’s `dt` setting. This function **does not resample**, making it preferable for chaotic or aperiodic systems. The major downside of this method is that it can does not have
  fixed points per period.

---

### Solver Choice

* All system integration is currently done using the **`RK45`** solver from `scipy.integrate.solve_ivp`. It offers an excellent balance between **accuracy and speed** for most **non-stiff, chaotic systems**.

* While more accurate solvers exist (e.g., `DOP853` or implicit solvers like `Radau`), they are typically **slower**. Interestingly, **lower-accuracy solvers** (or relaxed tolerances) can **introduce numerical noise**, which may act as **beneficial regularization** during training.

* For **stiff systems** — where some components change rapidly and others slowly — solvers like **`Radau`** or **`BDF`** become necessary. The default solver in Dysts is `Radau`, which is robust for stiff dynamics.

---

### Performance Observations

* For non-stiff systems like the Lorenz equations, **`RK45` is often faster than `RK23`**. Although each `RK45` step is more computationally intensive, it achieves **higher accuracy**, allowing it to take **larger steps** and **fewer total steps** overall.

* For stiff systems, however, this advantage diminishes or vanishes. `RK45` may suffer from frequent **step rejections** due to error control, making it less efficient. (This is a logical expectation; further empirical testing is needed.)

* The parameter_sweep is as optimized I can get it in this iteration. The GPU utilization saturates when the loop is calculating the  per batch and then takes a almost equivalent time to compute and allocating the next batch data. It was not very efficient because the main training part is the only one which is GPU bound. The weight allocation and tensor initialization takes a lot of time. and that happened each batch. But now with the method `build_all_weight()` the speed of each batch calculation increases significantly. But you are now GPU memory bound. Before there was no bound on maximum combination of the hyperparameter. After on 8gb card the maximum you can calculate is 1000 to 1300 combination. Which is still fine.
* As for the speed increase, the allocation takes 38.92 sec for 100 weights. but with the approximation of the eigen_values we can improve the allocation to 0.2 sec for 100 weights.
##### Performance Breakdown by Component

| Phase | Serial | Batching | Batching + PI | Speedup |
|-------|--------|----------|---------------|---------|
| **Model Init** | 0.26s/config | 0.475s/config | 0.007s/config | **37x** |
| **Training** | 1.88s/config | 0.245s/config | 0.178s/config | **10.6x** |
| **Prediction** | 2.45s/config | 0.304s/config | 0.152s/config | **16x** |
| **Weight Gen (N=1000-1500)** | N/A | 180-300s | 21s | **9-14x** |
| **Total (1000 configs)** | ~77 min | ~32 min | **~6.5 min** | **11.8x** |

## Core Idea

With the data generated, Now we can analyze each of the system at different points per period. The core idea is to find out if there exists
similar hyperparameter ranges where we expect similar results across 
different point per period. 

For that we then find a grid of the parameters, For our testing we have fixed the general grid. The core matrices to minimize or learn is matching trajectory. Which is basically minimizing Root mean square error.
We can see how the prediction is in a small time window and also 
if the system is learning long term dynamics with Lyapunov Exponent of 
the prediction vs true trajectory.

With the core idea out of the way we can discuss the implementation. 

## Implementation Details

1. Data Generation: The Lorenz and all the other attractors are generated using the `data_generation.py` script. It access, `systems.csv` file that has system name and type of the system. These are all the systems in the 
Dysts.


