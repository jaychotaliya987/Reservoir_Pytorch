# Study of Input Discretization Effects on Learning with Reservoir Computing

### Key Points

* The **Dysts dataset** includes an input discretization mechanism, typically tied to a dominant Fourier component or the Lyapunov exponent. This works well for **pseudo-chaotic** or **periodic systems**, where periods are well defined. However, in systems lacking a clear periodic structure, such resampling can **blur important dynamics** or introduce artifacts.

* I’ve implemented a function in `reservoirgrid.helpers.utils` to discretize **chaotic systems** by integrating them at a **fixed time step `dt`**. This approach avoids interpolating onto assumed periodic grids and respects the continuous chaotic dynamics more faithfully.

* There are key functions in `utils` for discretization:

  * `discretization`: Returns a NumPy array structured as `[[points_per_period], [data]]`, with the `data` aligned to the system's internal period estimates. Best suited for periodic or near-periodic systems.
  * `discretization_with_dt`: Returns just the `data` array, sampled at fixed intervals determined by the model’s `dt` setting. This function **does not resample**, making it preferable for chaotic or aperiodic systems.

---

### Solver Choice

* All system integration is currently done using the **`RK45`** solver from `scipy.integrate.solve_ivp`. It offers an excellent balance between **accuracy and speed** for most **non-stiff, chaotic systems**.

* While more accurate solvers exist (e.g., `DOP853` or implicit solvers like `Radau`), they are typically **slower**. Interestingly, **lower-accuracy solvers** (or relaxed tolerances) can **introduce numerical noise**, which may act as **beneficial regularization** during training.

* For **stiff systems** — where some components change rapidly and others slowly — solvers like **`Radau`** or **`BDF`** become necessary. The default solver in Dysts is `Radau`, which is robust for stiff dynamics.

---

### Performance Observations

* For non-stiff systems like the Lorenz equations, **`RK45` is often faster than `RK23`**. Although each `RK45` step is more computationally intensive, it achieves **higher accuracy**, allowing it to take **larger steps** and **fewer total steps** overall.

* For stiff systems, however, this advantage diminishes or vanishes. `RK45` may suffer from frequent **step rejections** due to error control, making it less efficient. (This is a logical expectation; further empirical testing is needed.)
