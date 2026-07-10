"""
Improved Bayesian Physics-Informed Neural Network (B-PINN) for Bergman Minimal Model
Using Pyro for variational inference with SIMULATED DATA for testing

Key Improvements:
1. Synthetic data generation with known parameters
2. Fixed physics loss computation during sampling
3. Validation/test split
4. Uncertainty quantification
5. Better normalization handling
6. Convergence diagnostics
7. Comprehensive posterior predictive checks

*** INSULIN DYNAMICS UPDATE ***
The insulin input now reproduces the two-component pharmacokinetic model
used in the real-data pipeline (ai_code.py):

  - Bolus (rapid-acting, e.g. Humulin R): biexponential absorption–elimination
        I_bolus(dt) = scaling * dose * F * [exp(-dt/tau_elim) - exp(-dt/tau_abs)]
        with peak normalization to expected_peak = scaling * dose * F * 0.4

  - Basal (long-acting, e.g. degludec): monoexponential approach-to-plateau
        I_basal(dt) = plateau * (1 - exp(-dt/tau_basal))
        where plateau = scaling_basal * dose * 0.7

  - Constant background offset: avg_basal_dose * 5.0  (mimics the
        circulating basal insulin present before any meal window)

  The Ib parameter passed to the ODE and the PINN model is set to 12.0 μU/mL,
  matching the Ib_estimate used in prepare_for_pinn() in ai_code.py.

Requirements:
pip install torch pyro-ppl matplotlib pandas numpy scipy
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.optim import ClippedAdam
from scipy.integrate import odeint
from scipy.stats import gaussian_kde

# ============================================================================
# CONFIGURATION — edit these
# ============================================================================
SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/bpinns-results/results_synthetic/"
FIX_P3 = False

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pyro.set_rng_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "lines.linewidth": 1.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PALETTE = {
    "p1": "#2166ac",  # blue
    "p2": "#d6604d",  # red-orange
    "p3": "#4dac26",  # green
    "obs": "#555555",
    "true": "#000000",
    "fill": 0.18,
    "grid": "#cccccc",
}


# ============================================================================
# 1. INSULIN PHARMACOKINETIC MODELS  (ported from ai_code.py)
# ============================================================================


def insulin_pk_bolus(t, t_dose, dose_IU, tau=50, bioavail=0.8):
    """
    Biexponential pharmacokinetic model for rapid-acting insulin (Humulin R).

    Reproduces the model in ai_code.py / insulin_pk_bolus().

    The plasma concentration profile is:
        I(dt) = scaling * dose * F * [exp(-dt/tau_elim) - exp(-dt/tau_abs)]
    rescaled so that the peak equals  scaling * dose * F * 0.4.

    Parameters
    ----------
    t        : array-like, time points (minutes)
    t_dose   : float, time of injection (minutes)
    dose_IU  : float, injected dose (IU)
    tau      : float, absorption time constant (minutes); default 50
    bioavail : float, subcutaneous bioavailability fraction; default 0.8

    Returns
    -------
    I : np.ndarray, plasma insulin contribution (μU/mL)
    """
    scaling_factor = 15.0

    dt = np.asarray(t, dtype=float) - t_dose
    I = np.zeros_like(dt)
    mask = dt >= 0

    tau_abs = tau
    tau_elim = tau * 3  # elimination is 3× slower than absorption

    I[mask] = (
        scaling_factor
        * dose_IU
        * bioavail
        * (np.exp(-dt[mask] / tau_elim) - np.exp(-dt[mask] / tau_abs))
    )

    # Normalise so that the actual peak equals the expected peak
    peak = I.max()
    if peak > 0:
        expected_peak = scaling_factor * dose_IU * bioavail * 0.4
        I = I * (expected_peak / peak)

    return I


def insulin_pk_basal(t, t_dose, dose_IU, tau=600):
    """
    Monoexponential saturation model for long-acting insulin (degludec).

    Reproduces the model in ai_code.py / insulin_pk_basal().

    The plasma concentration profile approaches a plateau:
        I(dt) = plateau * (1 - exp(-dt / tau_basal))
    where plateau = scaling_basal * dose * 0.7.

    Parameters
    ----------
    t        : array-like, time points (minutes)
    t_dose   : float, time of injection (minutes)
    dose_IU  : float, injected dose (IU)
    tau      : float, absorption time constant (minutes); default 600 (~10 h)

    Returns
    -------
    I : np.ndarray, plasma insulin contribution (μU/mL)
    """
    scaling_factor = 8.0

    dt = np.asarray(t, dtype=float) - t_dose
    I = np.zeros_like(dt)
    mask = dt >= 0

    plateau = scaling_factor * dose_IU * 0.7
    I[mask] = plateau * (1 - np.exp(-dt[mask] / tau))

    return I


def compute_insulin_profile(
    t_eval,
    bolus_events,  # list of (t_dose, dose_IU) tuples  — rapid-acting
    basal_events,  # list of (t_dose, dose_IU) tuples  — long-acting
    Ib=12.0,  # basal insulin background (μU/mL)
    basal_background_scale=5.0,  # matches avg_basal_dose * 5.0 in ai_code.py
):
    """
    Compute total plasma insulin concentration I(t) over t_eval.

    Mirrors compute_total_insulin() in ai_code.py:
      1. Constant background offset  = mean(basal_doses) * basal_background_scale
      2. Biexponential bolus contribution for every rapid-acting event
      3. Monoexponential plateau contribution for every long-acting event

    Parameters
    ----------
    t_eval               : 1-D array of time points (minutes)
    bolus_events         : list of (t_dose_min, dose_IU) for rapid-acting doses
    basal_events         : list of (t_dose_min, dose_IU) for long-acting doses
    Ib                   : float, basal insulin level added as constant offset
    basal_background_scale : float, scale applied to mean basal dose for the
                             constant background term (default 5.0)

    Returns
    -------
    I_total : np.ndarray, same shape as t_eval
    """
    I_total = np.zeros_like(t_eval, dtype=float)

    # ── constant background from long-acting insulin ──────────────────────────
    if basal_events:
        avg_basal_dose = np.mean([dose for _, dose in basal_events])
        I_total += avg_basal_dose * basal_background_scale

    # ── bolus contributions (biexponential) ───────────────────────────────────
    for t_dose, dose_IU in bolus_events:
        I_total += insulin_pk_bolus(t_eval, t_dose, dose_IU)

    # ── basal contributions (monoexponential plateau) ─────────────────────────
    for t_dose, dose_IU in basal_events:
        I_total += insulin_pk_basal(t_eval, t_dose, dose_IU)

    return I_total


# ============================================================================
# 2. SYNTHETIC DATA GENERATION
# ============================================================================


def bergman_odes(state, t, p1, p2, p3, I_func, Gb, Ib):
    """
    Bergman minimal model ODEs

    dG/dt = -p1*(G - Gb) - X*G
    dX/dt = -p2*X + p3*(I(t) - Ib)
    """
    G, X = state
    I = I_func(t)

    dG_dt = -p1 * (G - Gb) - X * G
    dX_dt = -p2 * X + p3 * (I - Ib)

    return [dG_dt, dX_dt]


def generate_synthetic_data(
    n_points=200,
    time_span=(0, 300),  # minutes  — mirrors a 4-h meal window
    true_params={"p1": 0.028, "p2": 0.025, "p3": 1.5e-5},
    Gb=150.0,  # mg/dL  — realistic T2DM pre-meal value
    Ib=12.0,  # μU/mL  — matches Ib_estimate in ai_code.py
    # ── Bolus events (rapid-acting insulin, e.g. Humulin R) ──────────────────
    bolus_times=[30.0],  # minutes  — single pre-meal bolus
    bolus_doses=[10.0],  # IU
    # ── Basal events (long-acting insulin, e.g. degludec) ────────────────────
    basal_times=[-120.0],  # minutes  — injected 2 h before window start
    basal_doses=[20.0],  # IU
    noise_std=5.0,  # mg/dL  — CGM measurement noise
    seed=42,
):
    """
    Generate synthetic diabetes data using the Bergman model with the same
    two-component insulin pharmacokinetic model used in ai_code.py.

    Key differences from the original glucose.py:
    - Bolus insulin uses insulin_pk_bolus() (biexponential PK, peak-normalised)
    - Basal insulin uses insulin_pk_basal() (monoexponential plateau)
    - A constant background offset from the basal dose is added
    - Default Gb / Ib reflect realistic clinical values (T2DM patient)
    - Default window is a single 4-h post-meal period with one bolus

    Returns
    -------
    dict with synthetic data and true parameters
    """
    np.random.seed(seed)

    print("=" * 70)
    print("GENERATING SYNTHETIC DATA  (real-data PK model)")
    print("=" * 70)
    print(f"\nTrue parameters:")
    print(f"  p1 = {true_params['p1']:.4f} min⁻¹ (glucose effectiveness)")
    print(f"  p2 = {true_params['p2']:.4f} min⁻¹ (insulin action decay)")
    print(f"  p3 = {true_params['p3']:.2e}  min⁻² per μU/mL (insulin sensitivity)")
    print(f"  Gb = {Gb:.1f} mg/dL (basal glucose)")
    print(f"  Ib = {Ib:.1f} μU/mL (basal insulin)")

    print(f"\nBolus events (rapid-acting, Humulin R PK):")
    for t_d, d in zip(bolus_times, bolus_doses):
        print(f"  t = {t_d:.0f} min  →  {d:.1f} IU")

    print(f"\nBasal events (long-acting, degludec PK):")
    for t_d, d in zip(basal_times, basal_doses):
        print(f"  t = {t_d:.0f} min  →  {d:.1f} IU")

    # ── time grid ─────────────────────────────────────────────────────────────
    t_eval = np.linspace(time_span[0], time_span[1], n_points)

    # ── build structured event lists ──────────────────────────────────────────
    bolus_events = list(zip(bolus_times, bolus_doses))
    basal_events = list(zip(basal_times, basal_doses))

    # ── compute insulin profile using real-data PK models ────────────────────
    I_true = compute_insulin_profile(t_eval, bolus_events, basal_events, Ib=Ib)

    # ── callable I(t) for the ODE solver (linear interpolation) ──────────────
    from scipy.interpolate import interp1d

    I_interp = interp1d(
        t_eval,
        I_true,
        kind="linear",
        bounds_error=False,
        fill_value=(I_true[0], I_true[-1]),
    )

    def insulin_function(t):
        return float(I_interp(t))

    # ── solve Bergman ODEs ────────────────────────────────────────────────────
    initial_state = [Gb, 0.0]  # start at basal glucose; X = 0

    solution = odeint(
        bergman_odes,
        initial_state,
        t_eval,
        args=(
            true_params["p1"],
            true_params["p2"],
            true_params["p3"],
            insulin_function,
            Gb,
            Ib,
        ),
    )

    G_true = solution[:, 0]
    X_true = solution[:, 1]

    # ── add CGM-like Gaussian noise ───────────────────────────────────────────
    G_obs = G_true + np.random.normal(0, noise_std, size=G_true.shape)

    print(f"\nGenerated data statistics:")
    print(f"  Time points : {n_points}")
    print(f"  Time range  : {t_eval[0]:.1f} – {t_eval[-1]:.1f} min")
    print(f"  G_true range: {G_true.min():.1f} – {G_true.max():.1f} mg/dL")
    print(f"  G_obs  range: {G_obs.min():.1f} – {G_obs.max():.1f} mg/dL")
    print(f"  I      range: {I_true.min():.2f} – {I_true.max():.2f} μU/mL")
    print(f"  X      range: {X_true.min():.5f} – {X_true.max():.5f}")
    print(f"  Noise σ     : {noise_std:.1f} mg/dL")

    return {
        "t_minutes": t_eval,
        "G_true": G_true,
        "G_obs": G_obs,
        "X_true": X_true,
        "I_true": I_true,
        "Gb": Gb,
        "Ib": Ib,
        "true_params": true_params,
        "noise_std": noise_std,
    }


def prepare_synthetic_data(synthetic_data):
    """
    Prepare synthetic data for PINN training.
    (Unchanged from original glucose.py — normalization logic is identical
     to ai_code.py / prepare_for_pinn().)
    """
    t_minutes = synthetic_data["t_minutes"]
    G_obs = synthetic_data["G_obs"]
    I_obs = synthetic_data["I_true"]
    Gb = synthetic_data["Gb"]
    Ib = synthetic_data["Ib"]

    # Normalization
    t_mean, t_std = t_minutes.mean(), max(t_minutes.std(), 1e-6)
    G_mean, G_std = G_obs.mean(), max(G_obs.std(), 1e-6)
    I_mean, I_std = I_obs.mean(), max(I_obs.std(), 1e-6)

    t_norm = (t_minutes - t_mean) / t_std
    G_norm = (G_obs - G_mean) / G_std
    I_norm = (I_obs - I_mean) / I_std

    data = {
        "t": torch.tensor(t_norm, dtype=torch.float32).reshape(-1, 1),
        "t_raw": torch.tensor(t_minutes, dtype=torch.float32).reshape(-1, 1),
        "G_obs": torch.tensor(G_norm, dtype=torch.float32),
        "G_obs_raw": torch.tensor(G_obs, dtype=torch.float32),
        "G_true_raw": torch.tensor(synthetic_data["G_true"], dtype=torch.float32),
        "X_true_raw": torch.tensor(synthetic_data["X_true"], dtype=torch.float32),
        "I_obs": torch.tensor(I_norm, dtype=torch.float32).reshape(-1, 1),
        "I_obs_raw": torch.tensor(I_obs, dtype=torch.float32).reshape(-1, 1),
        "Gb": float(Gb),
        "Ib": float(Ib),
        "normalization": {
            "t_mean": t_mean,
            "t_std": t_std,
            "G_mean": G_mean,
            "G_std": G_std,
            "I_mean": I_mean,
            "I_std": I_std,
        },
        "true_params": synthetic_data["true_params"],
        "noise_std": synthetic_data["noise_std"],
    }

    return data


def train_test_split(data, train_fraction=0.8, random=True):
    """Split data into training and test sets."""
    n = len(data["t"])
    n_train = int(n * train_fraction)
    indices = torch.randperm(n) if random else torch.arange(n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def split_dict(d, idx):
        return {
            k: (v[idx] if isinstance(v, torch.Tensor) and len(v) == n else v)
            for k, v in d.items()
        }

    print(f"\nData split:")
    print(f"  Training : {len(train_idx)} ({train_fraction*100:.0f}%)")
    print(f"  Test     : {len(test_idx)}  ({(1-train_fraction)*100:.0f}%)")

    return split_dict(data, train_idx), split_dict(data, test_idx)


# ============================================================================
# 3. NEURAL NETWORK  (unchanged)
# ============================================================================


class PINN(nn.Module):
    """Physics-Informed Neural Network for Bergman model"""

    def __init__(self, hidden_dims=[64, 64, 64], activation="tanh"):
        super().__init__()
        layers, input_dim = [], 1
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(
                {"tanh": nn.Tanh(), "relu": nn.ReLU(), "softplus": nn.Softplus()}.get(
                    activation, nn.Tanh()
                )
            )
            input_dim = h
        layers.append(nn.Linear(input_dim, 2))
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.network(t)


# ============================================================================
# 4. BAYESIAN MODEL AND GUIDE  (unchanged)
# ============================================================================


def model(
    t,
    G_obs,
    I_obs,
    Gb,
    Ib,
    norm_params,
    pinn_net,
    fix_p3=False,
    p3_value=1e-5,
    compute_physics=True,
):
    """Bayesian PINN model"""

    p1 = pyro.sample("p1", dist.LogNormal(torch.tensor(-3.58), torch.tensor(0.3)))
    p2 = pyro.sample("p2", dist.LogNormal(torch.tensor(-3.69), torch.tensor(0.3)))
    p3 = (
        torch.tensor(p3_value, dtype=torch.float32)
        if fix_p3
        else pyro.sample("p3", dist.LogNormal(torch.tensor(-11.1), torch.tensor(0.5)))
    )

    sigma_G = pyro.sample("sigma_G", dist.HalfNormal(0.5))
    sigma_phys = (
        pyro.sample("sigma_phys", dist.HalfNormal(0.2)) if compute_physics else None
    )

    pinn = pyro.module("pinn", pinn_net)

    if compute_physics:
        t_phys = t.clone().requires_grad_(True)
        out = pinn(t_phys)
        G_hat_norm = out[:, 0:1]
        X_hat_norm = out[:, 1:2]

        G_mean, G_std = norm_params["G_mean"], norm_params["G_std"]
        I_mean, I_std = norm_params["I_mean"], norm_params["I_std"]
        t_mean, t_std = norm_params["t_mean"], norm_params["t_std"]

        G_hat = G_hat_norm * G_std + G_mean
        X_hat = X_hat_norm
        I_phys = I_obs * I_std + I_mean

        dG_dt_norm = torch.autograd.grad(
            G_hat_norm,
            t_phys,
            torch.ones_like(G_hat_norm),
            create_graph=True,
            retain_graph=True,
        )[0]
        dX_dt_norm = torch.autograd.grad(
            X_hat_norm,
            t_phys,
            torch.ones_like(X_hat_norm),
            create_graph=True,
            retain_graph=True,
        )[0]

        dG_dt = dG_dt_norm * G_std / t_std
        dX_dt = dX_dt_norm / t_std

        R_G = dG_dt + p1 * (G_hat - Gb) + X_hat * G_hat
        R_X = dX_dt + p2 * X_hat - p3 * (I_phys - Ib)

        R_G_norm = R_G / G_std

        with pyro.plate("physics_plate", len(t)):
            pyro.sample(
                "physics_G", dist.Normal(0.0, sigma_phys), obs=R_G_norm.squeeze()
            )
            pyro.sample("physics_X", dist.Normal(0.0, sigma_phys), obs=R_X.squeeze())

        G_pred = G_hat_norm.squeeze()
    else:
        with torch.no_grad():
            G_pred = pinn(t)[:, 0]

    with pyro.plate("data_plate", len(G_obs)):
        pyro.sample("G_obs", dist.Normal(G_pred, sigma_G), obs=G_obs)


def guide(
    t,
    G_obs,
    I_obs,
    Gb,
    Ib,
    norm_params,
    pinn_net,
    fix_p3=False,
    p3_value=1e-5,
    compute_physics=True,
):
    """Variational guide"""

    p1_loc = pyro.param("mean_p1_loc", torch.tensor(-3.58))
    p1_scale = pyro.param(
        "mean_p1_scale", torch.tensor(0.3), constraint=dist.constraints.positive
    )
    p2_loc = pyro.param("mean_p2_loc", torch.tensor(-3.69))
    p2_scale = pyro.param(
        "mean_p2_scale", torch.tensor(0.3), constraint=dist.constraints.positive
    )

    pyro.sample("p1", dist.LogNormal(p1_loc, p1_scale))
    pyro.sample("p2", dist.LogNormal(p2_loc, p2_scale))

    if not fix_p3:
        p3_loc = pyro.param("mean_p3_loc", torch.tensor(-11.1))
        p3_scale = pyro.param(
            "mean_p3_scale", torch.tensor(0.5), constraint=dist.constraints.positive
        )
        pyro.sample("p3", dist.LogNormal(p3_loc, p3_scale))

    sG = pyro.param(
        "sigma_G_scale", torch.tensor(0.5), constraint=dist.constraints.positive
    )
    pyro.sample("sigma_G", dist.HalfNormal(sG))

    if compute_physics:
        sp = pyro.param(
            "sigma_phys_scale", torch.tensor(0.2), constraint=dist.constraints.positive
        )
        pyro.sample("sigma_phys", dist.HalfNormal(sp))

    pyro.module("pinn", pinn_net)


# ============================================================================
# 5. TRAINING  (unchanged from original glucose.py)
# ============================================================================


def check_convergence(losses, window=200, threshold=1e-4):
    """Check if training has converged."""
    if len(losses) < window:
        return False
    trend = np.polyfit(range(window), losses[-window:], 1)[0]
    return abs(trend) < threshold


def train_bayesian_pinn(
    train_data,
    val_data=None,
    n_iterations=5000,
    lr=0.001,
    fix_p3=False,
    save_dir="./checkpoints",
    checkpoint_freq=500,
    early_stopping_patience=1000,
):
    """Improved training with validation and early stopping."""
    print("\n" + "=" * 70)
    print("TRAINING BAYESIAN PINN")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)
    pyro.clear_param_store()

    pinn_net = PINN(hidden_dims=[64, 64, 64])
    optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    t, G_obs, I_obs = train_data["t"], train_data["G_obs"], train_data["I_obs"]
    Gb, Ib = train_data["Gb"], train_data["Ib"]
    norm_params = train_data["normalization"]

    train_losses, val_losses = [], []
    best_val_loss, patience = float("inf"), 0

    print(f"\n  Iterations : {n_iterations}")
    print(f"  LR         : {lr}")
    print(f"  Fix p3     : {fix_p3}")
    print(f"  Patience   : {early_stopping_patience}")

    for it in range(n_iterations):
        loss = svi.step(
            t, G_obs, I_obs, Gb, Ib, norm_params, pinn_net, fix_p3, 1e-5, True
        )
        train_losses.append(loss)

        if val_data is not None and (it + 1) % 50 == 0:
            vl = svi.evaluate_loss(
                val_data["t"],
                val_data["G_obs"],
                val_data["I_obs"],
                Gb,
                Ib,
                norm_params,
                pinn_net,
                fix_p3,
                1e-5,
                False,
            )
            val_losses.append(vl)

            if vl < best_val_loss:
                best_val_loss, patience = vl, 0
                torch.save(
                    {
                        "iteration": it + 1,
                        "train_loss": loss,
                        "val_loss": vl,
                        "pinn_state_dict": pinn_net.state_dict(),
                        "pyro_param_store": pyro.get_param_store().get_state(),
                    },
                    os.path.join(save_dir, "best_model.pt"),
                )
            else:
                patience += 50
                if patience >= early_stopping_patience:
                    print(f"\nEarly stopping at iteration {it + 1}")
                    break

        if (it + 1) % 100 == 0 or it == 0:
            msg = f"  Iter {it+1:5d}  train {loss:.3e}"
            if val_losses:
                msg += f"  val {val_losses[-1]:.3e}"
            print(msg)

        if (it + 1) % checkpoint_freq == 0:
            torch.save(
                {
                    "iteration": it + 1,
                    "train_loss": loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "pinn_state_dict": pinn_net.state_dict(),
                    "pyro_param_store": pyro.get_param_store().get_state(),
                },
                os.path.join(save_dir, f"checkpoint_{it+1}.pt"),
            )

        if (it + 1) % 500 == 0 and check_convergence(train_losses):
            print(f"\nConverged at iteration {it + 1}")
            break

    print("\nTraining completed!")
    return pinn_net, train_losses, val_losses


# ============================================================================
# 6. ANALYSIS, PREDICTION, AND VISUALIZATION
# ============================================================================


# ── HDI helper ───────────────────────────────────────────────────────────────
def hdi(s, cred=0.95):
    """Highest Density Interval."""
    s = np.sort(np.asarray(s).ravel())
    n = len(s)
    w = int(np.floor(cred * n))
    ws = s[w:] - s[: n - w]
    i = np.argmin(ws)
    return s[i], s[i + w]


# ── predict() — thin wrapper that normalises key names for the plot functions
def predict(net, data, samples, n=200):
    """
    Return a prediction dict with keys  mean / std / lower / upper
    (compatible with the publication plot functions).
    """
    unc = predict_with_uncertainty(net, data, samples, n_posterior_samples=n)
    return dict(
        mean=unc["G_mean"],
        std=unc["G_std"],
        lower=unc["G_lower"],
        upper=unc["G_upper"],
    )


def analyze_results(pinn_net, data, n_samples=1000, fix_p3=False):
    """Posterior analysis."""
    print("\n" + "=" * 70)
    print("POSTERIOR ANALYSIS")
    print("=" * 70)

    predictive = Predictive(model, guide=guide, num_samples=n_samples)
    samples = predictive(
        data["t"],
        data["G_obs"],
        data["I_obs"],
        data["Gb"],
        data["Ib"],
        data["normalization"],
        pinn_net,
        fix_p3,
        1e-5,
        False,
    )

    p1s = samples["p1"].detach().numpy()
    p2s = samples["p2"].detach().numpy()
    tp = data.get("true_params", {})

    print(
        f"\n{'Parameter':<10} {'Mean':<10} {'Std':<10} {'2.5%':<10} {'97.5%':<10} {'True':<10}"
    )
    print("-" * 60)
    print(
        f"{'p1':<10} {p1s.mean():<10.5f} {p1s.std():<10.5f} "
        f"{np.percentile(p1s, 2.5):<10.5f} {np.percentile(p1s, 97.5):<10.5f} "
        f"{tp.get('p1', 'N/A'):<10}"
    )
    print(
        f"{'p2':<10} {p2s.mean():<10.5f} {p2s.std():<10.5f} "
        f"{np.percentile(p2s, 2.5):<10.5f} {np.percentile(p2s, 97.5):<10.5f} "
        f"{tp.get('p2', 'N/A'):<10}"
    )

    if not fix_p3 and "p3" in samples:
        p3s = samples["p3"].detach().numpy()
        print(
            f"{'p3':<10} {p3s.mean():<10.6f} {p3s.std():<10.6f} "
            f"{np.percentile(p3s, 2.5):<10.6f} {np.percentile(p3s, 97.5):<10.6f} "
            f"{tp.get('p3', 'N/A'):<10}"
        )

    # Coverage
    if tp:
        p1_ok = np.percentile(p1s, 2.5) <= tp["p1"] <= np.percentile(p1s, 97.5)
        p2_ok = np.percentile(p2s, 2.5) <= tp["p2"] <= np.percentile(p2s, 97.5)
        print(
            f"\n95% CI Coverage:  p1={'✓' if p1_ok else '✗'}  p2={'✓' if p2_ok else '✗'}"
        )
        if not fix_p3 and "p3" in samples:
            p3_ok = np.percentile(p3s, 2.5) <= tp["p3"] <= np.percentile(p3s, 97.5)
            print(f"                  p3={'✓' if p3_ok else '✗'}")

    return samples


def predict_with_uncertainty(pinn_net, data, samples, n_posterior_samples=100):
    """Predictions with uncertainty quantification."""
    n_avail = len(samples["p1"])
    idx = np.random.choice(n_avail, min(n_posterior_samples, n_avail), replace=False)
    preds_G, preds_X = [], []

    with torch.no_grad():
        for i in idx:
            out = pinn_net(data["t"])
            preds_G.append(out[:, 0].numpy())
            preds_X.append(out[:, 1].numpy())

    preds_G = np.array(preds_G)
    preds_X = np.array(preds_X)

    norm = data["normalization"]
    preds_G = preds_G * norm["G_std"] + norm["G_mean"]

    return {
        "G_mean": preds_G.mean(0),
        "G_std": preds_G.std(0),
        "G_lower": np.percentile(preds_G, 2.5, 0),
        "G_upper": np.percentile(preds_G, 97.5, 0),
        "X_mean": preds_X.mean(0),
        "X_std": preds_X.std(0),
    }


def compute_metrics(data, predictions):
    """Performance metrics."""
    G_true = data["G_obs_raw"].numpy()
    G_pred = predictions["G_mean"]

    rmse = np.sqrt(np.mean((G_true - G_pred) ** 2))
    mae = np.mean(np.abs(G_true - G_pred))
    ss_res = np.sum((G_true - G_pred) ** 2)
    ss_tot = np.sum((G_true - G_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    cov = (
        np.mean((G_true >= predictions["G_lower"]) & (G_true <= predictions["G_upper"]))
        * 100
    )

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"  RMSE            : {rmse:.2f} mg/dL")
    print(f"  MAE             : {mae:.2f} mg/dL")
    print(f"  R²              : {r2:.4f}")
    print(f"  95% CI coverage : {cov:.1f}%")
    if "noise_std" in data:
        print(f"  RMSE / σ_noise  : {rmse/data['noise_std']:.2f}")


def plot_comprehensive_results(pinn_net, data, train_losses, val_losses, samples):
    """Comprehensive visualization"""

    fig = plt.figure(figsize=(18, 12))

    # Get predictions with uncertainty
    uncertainty = predict_with_uncertainty(pinn_net, data, samples)

    t_raw = data["t_raw"].numpy().flatten()
    G_obs_raw = data["G_obs_raw"].numpy()

    # Plot 1: Training curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label="Train", alpha=0.7)
    if val_losses:
        # Match validation losses to training iterations
        val_iters = np.linspace(0, len(train_losses), len(val_losses))
        ax1.plot(val_iters, val_losses, label="Validation", alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ELBO Loss")
    ax1.set_title("Training Progress")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Glucose predictions with uncertainty
    ax2 = plt.subplot(3, 3, 2)
    #ax2.fill_between(
    #    t_raw, uncertainty["G_lower"], uncertainty["G_upper"], alpha=0.5, label="95% CI"
    #)
    ax2.plot(t_raw, G_obs_raw, "o", alpha=0.5, markersize=3, label="Observed")
    if "G_true_raw" in data:
        ax2.plot(
            t_raw,
            data["G_true_raw"].numpy(),
            "--",
            linewidth=2,
            label="True",
            color="green",
        )
    ax2.plot(t_raw, uncertainty["G_mean"], "-", linewidth=0.5, label="Predicted")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Glucose (mg/dL)")
    ax2.set_title("Glucose: Predictions with Uncertainty")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax3 = plt.subplot(3, 3, 3)
    residuals = G_obs_raw - uncertainty["G_mean"]
    ax3.scatter(t_raw, residuals, alpha=0.5, s=10)
    ax3.axhline(y=0, color="r", linestyle="--")
    ax3.fill_between(
        t_raw,
        -2 * uncertainty["G_std"],
        2 * uncertainty["G_std"],
        alpha=0.2,
        color="gray",
    )
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Residual (mg/dL)")
    ax3.set_title("Prediction Residuals")
    ax3.grid(True, alpha=0.3)

    # Plot 4: X(t) trajectory with uncertainty
    ax4 = plt.subplot(3, 3, 4)
    ax4.fill_between(
        t_raw,
        uncertainty["X_mean"] - 2 * uncertainty["X_std"],
        uncertainty["X_mean"] + 2 * uncertainty["X_std"],
        alpha=0.3,
    )
    ax4.plot(t_raw, uncertainty["X_mean"], "-", linewidth=2, label="Predicted")
    if "X_true_raw" in data:
        ax4.plot(
            t_raw,
            data["X_true_raw"].numpy(),
            "--",
            linewidth=2,
            label="True",
            color="green",
        )
    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("Insulin Action X(t)")
    ax4.set_title("Insulin Action State")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Insulin input
    ax5 = plt.subplot(3, 3, 5)
    I_raw = data["I_obs_raw"].numpy().flatten()
    ax5.plot(t_raw, I_raw, linewidth=2)
    ax5.axhline(y=data["Ib"], color="r", linestyle="--", label=f"Ib = {data['Ib']:.1f}")
    ax5.set_xlabel("Time (minutes)")
    ax5.set_ylabel("Insulin (μU/mL)")
    ax5.set_title("Insulin Input")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Parameter posterior - p1
    ax6 = plt.subplot(3, 3, 6)
    p1_samples = samples["p1"].detach().numpy()
    ax6.hist(p1_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax6.axvline(
        p1_samples.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {p1_samples.mean():.4f}",
        linewidth=2,
    )
    if "true_params" in data:
        ax6.axvline(
            data["true_params"]["p1"],
            color="green",
            linestyle="-",
            label=f"True: {data['true_params']['p1']:.4f}",
            linewidth=2,
        )
    ax6.set_xlabel("p1 (min⁻¹)")
    ax6.set_ylabel("Density")
    ax6.set_title("Posterior: Glucose Effectiveness")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Parameter posterior - p2
    ax7 = plt.subplot(3, 3, 7)
    p2_samples = samples["p2"].detach().numpy()
    ax7.hist(p2_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax7.axvline(
        p2_samples.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {p2_samples.mean():.4f}",
        linewidth=2,
    )
    if "true_params" in data:
        ax7.axvline(
            data["true_params"]["p2"],
            color="green",
            linestyle="-",
            label=f"True: {data['true_params']['p2']:.4f}",
            linewidth=2,
        )
    ax7.set_xlabel("p2 (min⁻¹)")
    ax7.set_ylabel("Density")
    ax7.set_title("Posterior: Insulin Action Decay")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Parameter posterior - p3 or correlation
    ax8 = plt.subplot(3, 3, 8)
    if "p3" in samples:
        p3_samples = samples["p3"].detach().numpy()
        ax8.hist(p3_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
        ax8.axvline(
            p3_samples.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {p3_samples.mean():.6f}",
            linewidth=2,
        )
        if "true_params" in data:
            ax8.axvline(
                data["true_params"]["p3"],
                color="green",
                linestyle="-",
                label=f"True: {data['true_params']['p3']:.6f}",
                linewidth=2,
            )
        ax8.set_xlabel("p3 (min⁻² per μU/mL)")
        ax8.set_ylabel("Density")
        ax8.set_title("Posterior: Insulin Sensitivity")
        ax8.legend()
    else:
        # Show parameter correlation
        ax8.scatter(p1_samples, p2_samples, alpha=0.3, s=10)
        ax8.set_xlabel("p1")
        ax8.set_ylabel("p2")
        ax8.set_title("Parameter Correlation: p1 vs p2")

        # Add correlation coefficient
        corr = np.corrcoef(p1_samples, p2_samples)[0, 1]
        ax8.text(
            0.05,
            0.95,
            f"ρ = {corr:.3f}",
            transform=ax8.transAxes,
            verticalalignment="top",
        )
    ax8.grid(True, alpha=0.3)

    # Plot 9: Posterior predictive check
    ax9 = plt.subplot(3, 3, 9)
    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=ax9)
    ax9.set_title("Q-Q Plot: Residual Normality Check")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# 7. PUBLICATION PLOTS
# ============================================================================


def plot_training_curve(tr_losses, va_losses):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    ax.plot(tr_losses, color=PALETTE["p1"], lw=1.0, alpha=0.7, label="Training ELBO")
    if va_losses:
        vi = np.linspace(0, len(tr_losses), len(va_losses))
        ax.plot(vi, va_losses, color=PALETTE["p2"], lw=1.4, label="Validation ELBO")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, color=PALETTE["grid"], lw=0.5, ls="--", alpha=0.7)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, "bpinn_training_curve.pdf")
    fig.savefig(out, dpi=300)
    print(f"✓ Training curve  → {out}")
    plt.show()


def plot_glucose_fit(data, pred):
    t = data["t_raw"].numpy().flatten()
    G = data["G_obs_raw"].numpy()
    Gt = data["G_true_raw"].numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.5, 5.0),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    # ── top: fit ─────────────────────────────────────────────────────────────
    ax = axes[0]
    #ax.fill_between(
    #    t, pred["lower"], pred["upper"], color=PALETTE["p1"], alpha=0.20, label="95% PI"
    #)
    ax.plot(t, Gt, color=PALETTE["true"], lw=1.2, ls="--", label="True $G(t)$")
    ax.scatter(
        t,
        G,
        s=8,
        color=PALETTE["obs"],
        alpha=0.55,
        edgecolors="none",
        zorder=3,
        label="Observed",
    )
    ax.plot(t, pred["mean"], color=PALETTE["p1"], lw=1.6, label="B-PINN mean")

    ax.set_ylabel("Glucose (mg·dL$^{-1}$)")
    ax.legend(framealpha=0.9, handlelength=1.6, handletextpad=0.4, loc="upper right")
    ax.grid(True, color=PALETTE["grid"], lw=0.5, ls="--", alpha=0.7)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    # ── bottom: residuals ─────────────────────────────────────────────────────
    ax2 = axes[1]
    res = G - pred["mean"]
    ax2.scatter(t, res, s=6, color=PALETTE["obs"], alpha=0.55, edgecolors="none")
    ax2.axhline(0, color=PALETTE["true"], lw=0.9, ls="--")
    ax2.fill_between(
        t, -2 * pred["std"], 2 * pred["std"], color=PALETTE["p1"], alpha=0.18
    )
    ax2.set_ylabel("Residual\n(mg·dL$^{-1}$)", fontsize=9)
    ax2.set_xlabel("Time (min)")
    ax2.grid(True, color=PALETTE["grid"], lw=0.5, ls="--", alpha=0.7)
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, "bpinn_glucose_fit.pdf")
    fig.savefig(out, dpi=300)
    print(f"✓ Glucose fit     → {out}")
    plt.show()


def plot_marginal_posteriors(samples, true_params):
    param_meta = (
        [
            (
                "p1",
                r"$p_1$  (min$^{-1}$)",
                samples["p1"].detach().numpy(),
                PALETTE["p1"],
                true_params["p1"],
                1.0,
            ),
            (
                "p2",
                r"$p_2$  (min$^{-1}$)",
                samples["p2"].detach().numpy(),
                PALETTE["p2"],
                true_params["p2"],
                1.0,
            ),
            (
                "p3",
                r"$p_3$  ($\times 10^{-5}$)",
                samples["p3"].detach().numpy() * 1e5,
                PALETTE["p3"],
                true_params["p3"] * 1e5,
                1e5,
            ),
        ]
        if not FIX_P3
        else [
            (
                "p1",
                r"$p_1$  (min$^{-1}$)",
                samples["p1"].detach().numpy(),
                PALETTE["p1"],
                true_params["p1"],
                1.0,
            ),
            (
                "p2",
                r"$p_2$  (min$^{-1}$)",
                samples["p2"].detach().numpy(),
                PALETTE["p2"],
                true_params["p2"],
                1.0,
            ),
        ]
    )

    n_p = len(param_meta)
    fig, axes = plt.subplots(1, n_p, figsize=(3.0 * n_p, 2.8))
    fig.subplots_adjust(wspace=0.38)
    if n_p == 1:
        axes = [axes]

    for ax, (pname, xlabel, samp, color, true_val, _) in zip(axes, param_meta):
        samp = np.asarray(samp).ravel()
        kde = gaussian_kde(samp, bw_method="silverman")
        xs = np.linspace(
            samp.min() - 0.1 * (samp.max() - samp.min()),
            samp.max() + 0.1 * (samp.max() - samp.min()),
            2000,
        )
        ys = kde(xs)
        lo, hi = hdi(samp, 0.95)
        ymax = ys.max()

        # HDI fill
        mask = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs[mask], ys[mask], color=color, alpha=PALETTE["fill"])

        # KDE
        ax.plot(xs, ys, color=color, lw=1.8)

        # Mean & mode
        mean_ = samp.mean()
        mode_ = xs[np.argmax(ys)]
        ax.axvline(mean_, color=color, lw=1.2, ls="-", alpha=0.9)
        ax.axvline(mode_, color=color, lw=1.2, ls=":", alpha=0.7)

        # True value
        ax.axvline(true_val, color=PALETTE["true"], lw=1.4, ls="--", alpha=0.9)

        # HDI bracket
        ax.annotate(
            "",
            xy=(lo, ymax * 0.08),
            xytext=(hi, ymax * 0.08),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.0),
        )
        ax.text(
            (lo + hi) / 2,
            ymax * 0.13,
            "95% HDI",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
        )

        # Rug
        ax.plot(
            samp,
            np.full_like(samp, -ymax * 0.04),
            "|",
            color=color,
            alpha=0.45,
            ms=4,
            mew=0.8,
        )
        ax.set_ylim(-ymax * 0.08, None)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density" if pname == "p1" else "", fontsize=9)
        ax.set_title(f"$N={len(samp)}$ samples", fontsize=9, pad=3)
        ax.grid(True, color=PALETTE["grid"], lw=0.5, ls="--", alpha=0.7)

    # Shared legend
    legend_elems = [
        Line2D([0], [0], color="gray", lw=1.2, ls="-", label="Mean"),
        Line2D([0], [0], color="gray", lw=1.2, ls=":", label="Mode"),
        Line2D([0], [0], color="black", lw=1.4, ls="--", label="True value"),
        Patch(facecolor="gray", alpha=0.3, label="95% HDI"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=4,
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.10),
        columnspacing=1.2,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, "bpinn_posteriors.pdf")
    fig.savefig(out, dpi=300)
    print(f"✓ Posteriors      → {out}")
    plt.show()


def _to_1d(x):
    return np.asarray(x).reshape(-1)


def plot_corner(samples, true_params):
    labels = [
        (
            "p1",
            r"$p_1$ (min$^{-1}$)",
            samples["p1"].detach().numpy(),
            true_params["p1"],
            PALETTE["p1"],
        ),
        (
            "p2",
            r"$p_2$ (min$^{-1}$)",
            samples["p2"].detach().numpy(),
            true_params["p2"],
            PALETTE["p2"],
        ),
    ]
    if not FIX_P3:
        labels.append(
            (
                "p3",
                r"$p_3$ ($\times 10^{-5}$)",
                samples["p3"].detach().numpy() * 1e5,
                true_params["p3"] * 1e5,
                PALETTE["p3"],
            )
        )

    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.4 * n))
    fig.subplots_adjust(hspace=0.07, wspace=0.07)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            _, _, di, tv_i, ci = labels[i]
            _, _, dj, tv_j, cj = labels[j]

            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                di = np.asarray(di).ravel()
                kde = gaussian_kde(di, bw_method="silverman")
                xs = np.linspace(
                    di.min() - 0.05 * (di.max() - di.min()),
                    di.max() + 0.05 * (di.max() - di.min()),
                    1000,
                )
                ax.fill_between(xs, kde(xs), color=ci, alpha=0.22)
                ax.plot(xs, kde(xs), color=ci, lw=1.5)
                ax.axvline(tv_i, color=PALETTE["true"], lw=1.2, ls="--")
                ax.set_yticks([])
            else:
                dj = _to_1d(dj)
                di = _to_1d(di)

                mask = np.isfinite(dj) & np.isfinite(di)
                dj, di = dj[mask], di[mask]

                if len(dj) < 10:
                    continue

                xy = np.vstack([dj, di])
                kde2 = gaussian_kde(xy, bw_method="silverman")
                xg = np.linspace(dj.min(), dj.max(), 100)
                yg = np.linspace(di.min(), di.max(), 100)
                Xg, Yg = np.meshgrid(xg, yg)
                Zg = kde2(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

                lvls = sorted([Zg.max() * (1 - q) for q in [0.05, 0.30, 0.68, 0.95]])
                n_bands = len(lvls) - 1
                alphas = np.linspace(0.10, 0.45, n_bands)[::-1]
                cfill = [plt.matplotlib.colors.to_rgba(cj, a) for a in alphas]
                ax.contourf(Xg, Yg, Zg, levels=lvls, colors=cfill)
                ax.contour(
                    Xg, Yg, Zg, levels=lvls, colors=[cj], linewidths=0.7, alpha=0.8
                )

                ax.axvline(tv_j, color=PALETTE["true"], lw=0.9, ls="--", alpha=0.7)
                ax.axhline(tv_i, color=PALETTE["true"], lw=0.9, ls="--", alpha=0.7)

                rho = np.corrcoef(dj, di)[0, 1]
                ax.text(
                    0.96,
                    0.96,
                    f"$\\rho={rho:+.3f}$",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(
                        fc="white", ec="#aaaaaa", lw=0.5, boxstyle="round,pad=0.2"
                    ),
                )

            ax.grid(True, color=PALETTE["grid"], lw=0.4, ls="--", alpha=0.5)
            _, xlabel_j, _, _, _ = labels[j]
            _, xlabel_i, _, _, _ = labels[i]
            (
                ax.set_xlabel(xlabel_j, fontsize=9)
                if i == n - 1
                else plt.setp(ax.get_xticklabels(), visible=False)
            )
            (
                ax.set_ylabel(xlabel_i, fontsize=9)
                if j == 0 and i > 0
                else (
                    ax.set_ylabel(xlabel_i, fontsize=9)
                    if (j == 0 and i == 0)
                    else plt.setp(ax.get_yticklabels(), visible=False)
                )
            )
            ax.tick_params(labelsize=7.5)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, "bpinn_corner.pdf")
    fig.savefig(out, dpi=300)
    print(f"✓ Corner plot     → {out}")
    plt.show()


# ============================================================================
# 8. MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("BAYESIAN PINN — BERGMAN MODEL — SYNTHETIC DATA (real-data PK model)")
    print("=" * 70)

    N_ITERATIONS = 30_000
    LEARNING_RATE = 0.001
    CHECKPOINT_FREQ = 500
    EARLY_STOP_PATIENCE = 1000
    # SAVE_DIR and FIX_P3 are module-level configs — edit them at the top of the file.

    # ── Step 1: Generate synthetic data with real-data PK model ──────────────
    synthetic_data = generate_synthetic_data(
        n_points=200,
        time_span=(0, 300),  # 4-hour meal window (minutes)
        true_params={"p1": 0.028, "p2": 0.025, "p3": 1.5e-5},
        Gb=150.0,  # realistic T2DM pre-meal baseline
        Ib=12.0,  # matches Ib_estimate in ai_code.py
        # Single pre-meal bolus — typical outpatient scenario
        bolus_times=[30.0],
        bolus_doses=[10.0],
        # Basal dose injected 2 hours before the window starts
        basal_times=[-120.0],
        basal_doses=[20.0],
        noise_std=5.0,
        seed=42,
    )

    # ── Step 2: Prepare tensors ───────────────────────────────────────────────
    data = prepare_synthetic_data(synthetic_data)

    # ── Step 3: Train / test split ────────────────────────────────────────────
    train_data, test_data = train_test_split(data, train_fraction=0.8, random=True)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    pinn_net, train_losses, val_losses = train_bayesian_pinn(
        train_data,
        val_data=test_data,
        n_iterations=N_ITERATIONS,
        lr=LEARNING_RATE,
        fix_p3=FIX_P3,
        save_dir=SAVE_DIR,
        checkpoint_freq=CHECKPOINT_FREQ,
        early_stopping_patience=EARLY_STOP_PATIENCE,
    )

    # ── Step 5: Posterior analysis ────────────────────────────────────────────
    samples = analyze_results(pinn_net, data, n_samples=1000, fix_p3=FIX_P3)

    # ── Step 6: Predictions + metrics ─────────────────────────────────────────
    pred = predict(pinn_net, data, samples)
    compute_metrics(data, predict_with_uncertainty(pinn_net, data, samples))

    # ── Step 7: Publication plots ─────────────────────────────────────────────
    tp = data["true_params"]
    plot_training_curve(train_losses, val_losses)
    plot_glucose_fit(data, pred)
    plot_marginal_posteriors(samples, tp)
    plot_corner(samples, tp)

    print(f"\n✓ All outputs saved to: {SAVE_DIR}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()