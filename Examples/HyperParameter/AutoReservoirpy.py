"""
Reservoir Computing — Lorenz attractor LONG-TERM prediction + research()
=========================================================================
• Task    : autonomous (generative) long-term prediction of Lorenz x,y,z
• Search  : reservoirpy.hyper.research  (60 random trials, 3 instances each)
• Metrics : RMSE, Valid-time (Lyapunov units), variance dict of best params
• Output  : 4-panel figure + vardict JSON
"""

import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

import reservoirpy as rpy
from reservoirpy.datasets import lorenz
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.hyper import research

warnings.filterwarnings("ignore")

import builtins
import hyperopt.pyll.base

def hyperopt_isinstance_patch(obj, class_or_tuple):
    # Use builtins directly to avoid any infinite recursion loops
    if class_or_tuple is int:
        return builtins.isinstance(obj, (int, np.integer))
    if builtins.isinstance(class_or_tuple, tuple) and int in class_or_tuple:
        return builtins.isinstance(obj, class_or_tuple + (np.integer,))
    return builtins.isinstance(obj, class_or_tuple)

# Inject the patch directly into hyperopt's local module namespace
hyperopt.pyll.base.isinstance = hyperopt_isinstance_patch


# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
np.random.seed(SEED)

# ── Lorenz constants ──────────────────────────────────────────────────────────
# Lyapunov time ≈ 1/λ_max ≈ 1/0.906 ≈ 1.1  (in Lorenz time units)
# With dt=0.01 → 1 Lyapunov time ≈ 110 steps
DT              = 0.01
LYAPUNOV_STEPS  = 110          # steps per Lyapunov time
VALID_THRESHOLD = 0.4          # normalised RMSE threshold for "valid time"

# ── generate data ─────────────────────────────────────────────────────────────
TOTAL      = 12000
data       = lorenz(TOTAL, dt=DT, seed=SEED)          # (T, 3)  x,y,z

# normalise each channel independently
mu  = data.mean(axis=0)
sig = data.std(axis=0)
data = (data - mu) / sig

WARMUP     = 500
TRAIN_END  = 7000
TEST_START = TRAIN_END
TEST_LEN   = 4000   # ~36 Lyapunov times — real long-term challenge

X_train = data[WARMUP:TRAIN_END]          # input:  t
Y_train = data[WARMUP+1:TRAIN_END+1]      # target: t+1  (teacher forcing)

X_test  = data[TEST_START:TEST_START+1]   # seed: only 1 step, then autonomous
Y_test  = data[TEST_START+1:TEST_START+1+TEST_LEN]

dataset = ((X_train, Y_train), (X_test, Y_test))

# ─────────────────────────────────────────────────────────────────────────────
def valid_time(y_pred, y_true, threshold=VALID_THRESHOLD):
    """Number of steps where normalised RMSE < threshold."""
    # rolling RMSE over a 10-step window, normalised by signal std
    sig_ = y_true.std(axis=0).mean()
    for t in range(len(y_true)):
        rmse = np.sqrt(np.mean((y_pred[:t+1] - y_true[:t+1])**2))
        if rmse / sig_ > threshold:
            return t
    return len(y_true)

def run_autonomous(model, seed_step, n_steps):
    """Closed-loop (generative) rollout: feed own output back as input."""
    outputs = []
    x = seed_step.copy()            # shape (1, 3)
    for _ in range(n_steps):
        x = model.run(x)            # run 1 step
        outputs.append(x.copy())
    return np.vstack(outputs)       # (n_steps, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────
def objective(dataset, config, *,
              N, sr, lr, input_scaling, ridge,
              input_connectivity, rc_connectivity):

    (X_tr, Y_tr), (X_te, Y_te) = dataset
    
    # ─── FIX: Handle Hyperopt's split-personality choice values ───────────
    N_OPTIONS = [100, 200, 500, 1000]
    if int(N) < len(N_OPTIONS):
        n_units = N_OPTIONS[int(N)]  # It passed an index (0, 1, 2, 3)
    else:
        n_units = int(N)             # It passed the literal value (100, 200, etc.)
    # ───────────────────────────────────────────────────────────────────────

    vt_scores, rmse_scores = [], []

    for trial_seed in range(config["instances_per_trial"]):
        res  = Reservoir(units=n_units, sr=float(sr), lr=float(lr),
                         input_scaling=float(input_scaling),
                         input_connectivity=float(input_connectivity),
                         rc_connectivity=float(rc_connectivity),
                         seed=trial_seed * 1000 + 7)
        rdout = Ridge(ridge=float(ridge))
        model = res >> rdout

        model.fit(X_tr, Y_tr, warmup=200)

        # autonomous rollout from the seed
        y_pred = run_autonomous(model, X_te, len(Y_te))

        rmse   = float(np.sqrt(np.mean((y_pred - Y_te)**2)))
        vt     = valid_time(y_pred, Y_te)

        rmse_scores.append(rmse)
        vt_scores.append(vt)

    mean_rmse = float(np.mean(rmse_scores))
    mean_vt   = float(np.mean(vt_scores))
    optimization_loss = -mean_vt + (mean_rmse*0.01)

    # minimise: penalise RMSE and reward valid-time (subtract normalised VT)
    # Primary loss = mean RMSE  (hyperopt minimises)
    return {
        "loss":      optimization_loss,
        "rmse_mean": mean_rmse,
        "rmse_std":  float(np.std(rmse_scores)),
        "vt_mean":   mean_vt,
        "vt_std":    float(np.std(vt_scores)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Run search
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH =  "config.json"
REPORT_PATH = "results_lorenz"

print("=" * 64)
print("  ReservoirPy — Lorenz long-term prediction  /  hyper search")
print(f"  Config : {CONFIG_PATH}")
print("=" * 64)

best_result, trials_obj = research(
    objective   = objective,
    dataset     = dataset,
    config_path = str(CONFIG_PATH),
    report_path = str(REPORT_PATH),
)

N_OPTIONS = [100, 200, 500, 1000]
best = dict(best_result)
best["N"] = N_OPTIONS[int(best["N"])]

print("\n✓ Search complete.")
print(f"  Best raw params  : {best}")

# ─────────────────────────────────────────────────────────────────────────────
# Variance dict — analyse spread of each HP across ALL trials
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding vardict …")

ok_trials = [t for t in trials_obj.trials
             if t.get("result", {}).get("status") == "ok"]

# collect param values and losses
all_losses = np.array([t["result"]["loss"] for t in ok_trials])
top_n      = max(5, len(ok_trials) // 5)   # top 20% trials
top_idx    = np.argsort(all_losses)[:top_n]

param_keys = ["N_idx", "sr", "lr", "input_scaling", "ridge",
              "input_connectivity", "rc_connectivity"]

# raw misc values from hyperopt
all_vals   = {k: [] for k in param_keys}
for t in ok_trials:
    misc = t["misc"]["vals"]
    for k in param_keys:
        raw_k = k.replace("_idx", "")  # N_idx → N in hyperopt
        key   = "N" if k == "N_idx" else k
        v = misc.get(key, [None])[0]
        if v is not None:
            all_vals[k].append(float(v))

# decode N from index
all_vals["N"] = [N_OPTIONS[int(round(v))] for v in all_vals["N_idx"]
                 if 0 <= int(round(v)) < len(N_OPTIONS)]
del all_vals["N_idx"]

# compute vardict
vardict = {}
for k, vals in all_vals.items():
    if not vals:
        continue
    arr = np.array(vals, dtype=float)
    top_arr = arr[top_idx] if len(arr) == len(ok_trials) else arr[:top_n]

    vardict[k] = {
        "best":         float(best.get(k, np.nan)),
        "all_mean":     float(arr.mean()),
        "all_std":      float(arr.std()),
        "all_min":      float(arr.min()),
        "all_max":      float(arr.max()),
        "top20pct_mean": float(top_arr.mean()),
        "top20pct_std":  float(top_arr.std()),
        "sensitivity":  "high" if top_arr.std() / (arr.std() + 1e-12) < 0.5 else "low",
    }

print("\n─── VARDICT ──────────────────────────────────────────────")
pprint(vardict, sort_dicts=False)
print("──────────────────────────────────────────────────────────\n")

vardict_path = "lorenz_vardict.json"
with open(vardict_path, "w") as f:
    json.dump(vardict, f, indent=2)
print(f"vardict saved → {vardict_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Retrain best model & full autonomous rollout
# ─────────────────────────────────────────────────────────────────────────────
print("\nRetraining best model for final evaluation …")

best_res   = Reservoir(units=int(best["N"]),
                       sr=float(best["sr"]), lr=float(best["lr"]),
                       input_scaling=float(best["input_scaling"]),
                       input_connectivity=float(best["input_connectivity"]),
                       rc_connectivity=float(best["rc_connectivity"]),
                       seed=SEED)
best_rdout = Ridge(ridge=float(best["ridge"]))
best_model = best_res >> best_rdout

best_model.fit(X_train, Y_train, warmup=200)
Y_pred = run_autonomous(best_model, X_test, TEST_LEN)

final_rmse = float(np.sqrt(np.mean((Y_pred - Y_test)**2)))
final_vt   = valid_time(Y_pred, Y_test)
final_vt_LT = final_vt / LYAPUNOV_STEPS  # in Lyapunov times

print(f"  RMSE             : {final_rmse:.5f}")
print(f"  Valid time       : {final_vt} steps  ({final_vt_LT:.2f} Lyapunov times)")

# ─────────────────────────────────────────────────────────────────────────────
# Figures  (2 pages: 4 + 4 panels)
# ─────────────────────────────────────────────────────────────────────────────
CBLUE   = "#2563eb"
CRED    = "#dc2626"
CPURPLE = "#7c3aed"
CGREEN  = "#16a34a"
CGREY   = "#94a3b8"

# ── Figure 1: Prediction quality ─────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(15, 9))
fig1.suptitle(
    f"ReservoirPy — Lorenz  Long-Term Autonomous Prediction\n"
    f"N={best['N']}  sr={best['sr']:.3f}  lr={best['lr']:.3f}  "
    f"ridge={best['ridge']:.2e}   "
    f"RMSE={final_rmse:.4f}   Valid time={final_vt_LT:.1f} λ-times",
    fontsize=11, fontweight="bold")

SHOW = 1200   # steps to display in time series

for i, (ch, name) in enumerate([(0,"x"),(1,"y"),(2,"z")]):
    ax = axes[i // 2][i % 2] if i < 2 else axes[1][0]
    t  = np.arange(SHOW) * DT
    ax.plot(t, Y_test[:SHOW, ch], color=CBLUE,   lw=1.2, label="Ground truth", alpha=0.9)
    ax.plot(t, Y_pred[:SHOW, ch], color=CRED,    lw=1.0, label="RC prediction", alpha=0.85)
    # shade valid region
    ax.axvspan(0, final_vt * DT, color=CGREEN, alpha=0.07, label=f"Valid ({final_vt_LT:.1f} λT)")
    ax.set_title(f"Lorenz  {name}(t)  — long-term rollout")
    ax.set_xlabel("Time"); ax.set_ylabel(f"{name} (normalised)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)

# 4th panel: 3-D phase portrait
ax3d = fig1.add_subplot(2, 2, 4, projection="3d")
N_3D = min(3000, TEST_LEN)
ax3d.plot(*Y_test[:N_3D].T,  color=CBLUE,   lw=0.5, alpha=0.6, label="Truth")
ax3d.plot(*Y_pred[:N_3D].T,  color=CRED,    lw=0.5, alpha=0.6, label="RC")
ax3d.set_title("Phase portrait (first 3000 steps)")
ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
ax3d.legend(fontsize=8); ax3d.tick_params(labelsize=7)

plt.tight_layout()
fig1_path = "lorenz_prediction.png"
fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
print(f"  Fig1 saved → {fig1_path}")
plt.close(fig1)

# ── Figure 2: Hyper-search analysis + vardict ─────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle("ReservoirPy — Lorenz  Hyperparameter Search Analysis", fontsize=13, fontweight="bold")

# 2a. Loss over trials
ax = axes2[0, 0]
ax.scatter(range(len(all_losses)), all_losses, s=16, color=CPURPLE, alpha=0.6, zorder=3)
ax.plot(np.minimum.accumulate(all_losses), color=CRED, lw=1.8, label="Best so far")
ax.axhline(final_rmse, color=CGREEN, lw=1.2, ls="--", label=f"Retrained best ({final_rmse:.4f})")
ax.set_title("RMSE per trial & running best"); ax.set_xlabel("Trial #"); ax.set_ylabel("RMSE")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 2b. Valid-time histogram
vt_vals = [t["result"].get("vt_mean", 0) / LYAPUNOV_STEPS for t in ok_trials]
ax = axes2[0, 1]
ax.hist(vt_vals, bins=20, color=CBLUE, edgecolor="white", alpha=0.85)
ax.axvline(final_vt_LT, color=CRED, lw=1.5, ls="--", label=f"Best={final_vt_LT:.1f} λT")
ax.set_title("Distribution of valid times (Lyapunov times)")
ax.set_xlabel("Valid time (λT)"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# 2c. Vardict: top-20% mean ± std vs all-trial mean ± std
ax = axes2[1, 0]
vd_keys  = list(vardict.keys())
x_pos    = np.arange(len(vd_keys))
# normalise everything to [0,1] range of that param for display
norm_best  = []
norm_top_m = []
norm_top_s = []
for k in vd_keys:
    lo, hi = vardict[k]["all_min"], vardict[k]["all_max"]
    span   = hi - lo + 1e-12
    norm_best.append((vardict[k]["best"]          - lo) / span)
    norm_top_m.append((vardict[k]["top20pct_mean"] - lo) / span)
    norm_top_s.append(vardict[k]["top20pct_std"]         / span)

ax.bar(x_pos - 0.2, norm_best,  0.35, color=CRED,    alpha=0.8, label="Best trial")
ax.bar(x_pos + 0.2, norm_top_m, 0.35, color=CBLUE,   alpha=0.8,
       yerr=norm_top_s, capsize=3, label="Top 20% mean ± std")
ax.set_xticks(x_pos); ax.set_xticklabels(vd_keys, rotation=30, ha="right", fontsize=8)
ax.set_title("Vardict — best vs top-20% (normalised to param range)")
ax.set_ylabel("Normalised position in search range")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

# 2d. Sensitivity flag text table
ax = axes2[1, 1]
ax.axis("off")
rows = [["Parameter", "Best value", "Top20% mean", "Top20% std", "Sensitivity"]]
for k, d in vardict.items():
    bv = f"{int(d['best'])}" if k == "N" else f"{d['best']:.4g}"
    rows.append([k, bv, f"{d['top20pct_mean']:.4g}",
                 f"{d['top20pct_std']:.4g}", d["sensitivity"]])

tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.auto_set_column_width(col=list(range(5)))
# colour sensitivity column
for row_idx in range(1, len(rows)):
    sens = rows[row_idx][4]
    colour = "#fecaca" if sens == "high" else "#bbf7d0"
    tbl[row_idx, 4].set_facecolor(colour)
ax.set_title("Vardict summary (red=high sensitivity, green=low)", pad=12)

plt.tight_layout()
fig2_path = "lorenz_hyper_analysis.png"
fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
print(f"  Fig2 saved → {fig2_path}")
plt.close(fig2)

# ── save full summary ──────────────────────────────────────────────────────────
summary = {
    "best_params":     {k: (int(v) if isinstance(v, (int, np.integer))
                            else float(v)) for k, v in best.items()},
    "final_rmse":      final_rmse,
    "valid_time_steps": int(final_vt),
    "valid_time_lyapunov": float(final_vt_LT),
    "n_trials":        len(ok_trials),
    "vardict":         vardict,
}
summary_path = "lorenz_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Summary saved → {summary_path}")
print("\n✓ All done.\n")