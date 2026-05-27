"""
Benchmark: ReservoirGrid (GPU-batched) vs ReservoirPy (CPU-sequential)
======================================================================
Same dataset, same metric, timed end-to-end.
Run this from the ReservoirGrid root directory.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from reservoirgrid.helpers import utils
from reservoirgrid.helpers import viz
from reservoirgrid.models.Reservoir import Reservoir

# ── Shared data ───────────────────────────────────────────────────────────────
dataset = np.load("reservoirgrid/datasets/Chaotic/Lorenz.npy", allow_pickle=True)
dataset = utils.normalize_data(dataset[15][1])
X_train, X_val, Y_train, Y_val = utils.split(dataset)

def to_numpy(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.array(x)

X_train_np = to_numpy(X_train)
Y_train_np = to_numpy(Y_train)
X_val_np   = to_numpy(X_val)
Y_val_np   = to_numpy(Y_val)

results = {}

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — ReservoirGrid  (GPU-batched, Optuna)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  METHOD 1 — ReservoirGrid (your method)")
print("="*60)

t0 = time.perf_counter()

rg_model = Reservoir(input_dim=3, reservoir_dim=1000, output_dim=3)
rg_best  = rg_model.optimize(
    X_train=X_train, Y_train=Y_train,
    X_val=X_val,     Y_val=Y_val,
    metric_fn=utils.RMSE,
    direction="minimize",
    n_trials=75,
    batch_size=3,
)

rg_preds = rg_model.predict(initial_input=Y_train, steps=len(Y_val))
rg_preds_np = to_numpy(rg_preds)
Y_val_np   = to_numpy(Y_val)

rg_time = time.perf_counter() - t0
rg_rmse = float(utils.RMSE(Y_val_np, rg_preds_np))

results["ReservoirGrid"] = {"rmse": rg_rmse, "time": rg_time, "params": rg_best}

print(f"\n  RMSE : {rg_rmse:.5f}")
print(f"  Time : {rg_time:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — ReservoirPy  (CPU-sequential, hyperopt)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  METHOD 2 — ReservoirPy (baseline)")
print("="*60)

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir as RPyReservoir, Ridge
from reservoirpy.hyper import research
rpy.set_seed(42)

CONFIG_PATH = "Examples/HyperParameter/config.json"
REPORT_PATH = "Examples/HyperParameter/results_lorenz_bench"

def rpy_objective(dataset, config, *, N, sr, lr, input_scaling, ridge,
                  input_connectivity, rc_connectivity):
    (X_tr, Y_tr), (X_te, Y_te) = dataset
    n_units = int(round(float(N)))
    losses = []
    for seed in range(config["instances_per_trial"]):
        res   = RPyReservoir(units=n_units, sr=float(sr), lr=float(lr),
                             input_scaling=float(input_scaling),
                             input_connectivity=float(input_connectivity),
                             rc_connectivity=float(rc_connectivity),
                             seed=seed * 1000 + 7)
        model = res >> Ridge(ridge=float(ridge))
        model.fit(X_tr, Y_tr, warmup=200)
        x = X_te[:1].copy()
        preds = []
        for _ in range(len(Y_te)):
            x = model.run(x)
            preds.append(x.copy())
        rmse = float(np.sqrt(np.mean((np.vstack(preds) - Y_te) ** 2)))
        losses.append(rmse)
    return {"loss": float(np.mean(losses)), "status": "ok"}

t0 = time.perf_counter()

rpy_best, _ = research(
    objective   = rpy_objective,
    dataset     = ((X_train_np, Y_train_np), (X_val_np, Y_val_np)),
    config_path = CONFIG_PATH,
    report_path = REPORT_PATH,
)
rpy_best["N"] = int(round(float(rpy_best["N"])))

rpy_res   = RPyReservoir(units=rpy_best["N"],
                         sr=float(rpy_best["sr"]),
                         lr=float(rpy_best["lr"]),
                         input_scaling=float(rpy_best["input_scaling"]),
                         input_connectivity=float(rpy_best["input_connectivity"]),
                         rc_connectivity=float(rpy_best["rc_connectivity"]),
                         seed=42)
rpy_model = rpy_res >> Ridge(ridge=float(rpy_best["ridge"]))
rpy_model.fit(X_train_np, Y_train_np, warmup=200)

x = X_val_np[:1].copy()
rpy_preds = []
for _ in range(len(Y_val_np)):
    x = rpy_model.run(x)
    rpy_preds.append(x.copy())
rpy_preds_np = np.vstack(rpy_preds)

rpy_time = time.perf_counter() - t0
rpy_rmse = float(np.sqrt(np.mean((rpy_preds_np - Y_val_np) ** 2)))

results["ReservoirPy"] = {"rmse": rpy_rmse, "time": rpy_time, "params": rpy_best}

print(f"\n  RMSE : {rpy_rmse:.5f}")
print(f"  Time : {rpy_time:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  BENCHMARK SUMMARY")
print("="*60)
rg = results["ReservoirGrid"]
rp = results["ReservoirPy"]
speedup      = rp["time"] / rg["time"]
rmse_improve = (rp["rmse"] - rg["rmse"]) / rp["rmse"] * 100

print(f"  {'Method':<20} {'RMSE':>10} {'Time (s)':>12}")
print(f"  {'-'*44}")
print(f"  {'ReservoirGrid':<20} {rg['rmse']:>10.5f} {rg['time']:>11.1f}s")
print(f"  {'ReservoirPy':<20} {rp['rmse']:>10.5f} {rp['time']:>11.1f}s")
print(f"  {'-'*44}")
print(f"  Speedup      : {speedup:.1f}x faster")
print(f"  RMSE improve : {rmse_improve:+.1f}%")
print("="*60 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Plots — compare both predictions against ground truth
# ═══════════════════════════════════════════════════════════════════════════════
viz.compare_components(
    [Y_val_np, rg_preds_np, rpy_preds_np],
    labels=["True", "ReservoirGrid", "ReservoirPy"],
    title="Benchmark — ReservoirGrid vs ReservoirPy"
).show()

viz.compare_plot(
    [Y_val_np, rg_preds_np, rpy_preds_np],
    labels=["True", "ReservoirGrid", "ReservoirPy"],
    title="Benchmark — ReservoirGrid vs ReservoirPy"
).show()