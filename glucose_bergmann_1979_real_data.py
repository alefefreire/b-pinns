"""
Complete Bayesian PINN Analysis for Patient Glucose Data

This script integrates:
1. Data preparation from prepare_patient_data_for_pinn.py
2. B-PINN training from improved_bpinn_synthetic.py
3. Comprehensive analysis and visualization

Usage:
------
    Simply modify the configuration variables below and run:
    python run_bpinn_analysis.py

Configuration:
--------------
    Edit the variables in the CONFIGURATION section below
"""

import os
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.optim import ClippedAdam
from scipy.interpolate import interp1d

# ============================================================================
# CONFIGURATION - EDIT THESE VARIABLES TO CUSTOMIZE YOUR ANALYSIS
# ============================================================================

# Which meal window to analyze (0-39 available for patient 1006)
# Window 0: Normal baseline (184 mg/dL), good starting point
# Window 4: High baseline (286 mg/dL), poor control
# Window 5: Low baseline (69 mg/dL), hypoglycemic
WINDOW = 0

# Training parameters
ITERATIONS = 20000  # Number of training iterations (5000=quick test, 20000=standard, 50000=high precision)
LEARNING_RATE = (
    0.001  # Learning rate for optimization (0.001=standard, 0.0005=more stable)
)
FIX_P3 = False  # Fix p3 parameter? (True=faster/more stable, False=infer from data)

# Run on multiple representative windows instead of single window?
RUN_ALL_WINDOWS = True  # True=analyze ALL available windows, False=analyze WINDOW only

# Data file path
DATA_FILE = "/content/1011_0_20210622.xls"

# Results output directory
RESULTS_DIR = "/content/drive/MyDrive/Colab Notebooks/bpinns-results/shangai/1011_0_20210622"

# Advanced options
TRAIN_FRACTION = 0.8  # Fraction of data for training (rest for validation)
EARLY_STOPPING_PATIENCE = 3000  # Stop if no improvement for this many iterations
POSTERIOR_SAMPLES = 1000  # Number of posterior samples for uncertainty quantification
NETWORK_HIDDEN_DIMS = [64, 64, 64]  # Neural network architecture

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"\nConfiguration:")
print(f"  Window: {WINDOW}")
print(f"  Iterations: {ITERATIONS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Fix p3: {FIX_P3}")
print(f"  Run all windows: {RUN_ALL_WINDOWS}")
print(f"  Data file: {DATA_FILE}")
print(f"  Results directory: {RESULTS_DIR}")


# ============================================================================
# PART 1: DATA PREPARATION FUNCTIONS
# ============================================================================


def extract_insulin_dose(text):
    """Extract insulin dose and type from text"""
    if pd.isna(text):
        return None, None

    match = re.search(r"(\d+(?:\.\d+)?)\s*IU", str(text))
    if match:
        dose = float(match.group(1))
        if "degludec" in text.lower() or "tresiba" in text.lower():
            insulin_type = "basal"
        elif "humulin" in text.lower() or "novolin" in text.lower():
            insulin_type = "bolus"
        else:
            insulin_type = "unknown"
        return dose, insulin_type
    return None, None


def insulin_pk_bolus(t, t_dose, dose_IU, tau=50, bioavail=0.8):
    """
    Pharmacokinetic model for rapid-acting insulin (Humulin R)
    """
    scaling_factor = 15.0

    dt = t - t_dose
    I = np.zeros_like(dt)
    mask = dt >= 0

    tau_abs = tau
    tau_elim = tau * 3

    I[mask] = (
        scaling_factor
        * dose_IU
        * bioavail
        * (np.exp(-dt[mask] / tau_elim) - np.exp(-dt[mask] / tau_abs))
    )

    peak = I.max()
    if peak > 0:
        expected_peak = scaling_factor * dose_IU * bioavail * 0.4
        I = I * (expected_peak / peak)

    return I


def insulin_pk_basal(t, t_dose, dose_IU, tau=600):
    """
    Pharmacokinetic model for long-acting insulin (degludec)
    """
    scaling_factor = 8.0

    dt = t - t_dose
    I = np.zeros_like(dt)
    mask = dt >= 0

    plateau = scaling_factor * dose_IU * 0.7
    I[mask] = plateau * (1 - np.exp(-dt[mask] / tau))

    return I


def compute_total_insulin(df, insulin_events):
    """Compute total plasma insulin concentration time series"""
    t = df["Time_minutes"].values
    I_total = np.zeros_like(t)

    basal_events = insulin_events[insulin_events["type"] == "basal"]
    if len(basal_events) > 0:
        avg_basal_dose = basal_events["dose_IU"].mean()
        basal_contribution = avg_basal_dose * 5.0
        I_total += basal_contribution

    bolus_events = insulin_events[insulin_events["type"] == "bolus"]
    for _, event in bolus_events.iterrows():
        I_bolus = insulin_pk_bolus(t, event["time_minutes"], event["dose_IU"])
        I_total += I_bolus

    for _, event in basal_events.iterrows():
        I_basal = insulin_pk_basal(t, event["time_minutes"], event["dose_IU"])
        I_total += I_basal

    return I_total


def extract_meal_windows(
    df, insulin_events, meal_times, window_hours=4, min_gap_hours=2
):
    """Extract individual meal-response windows for cleaner PINN analysis"""
    windows = []
    last_selected = -np.inf

    for meal_time in meal_times:
        if (meal_time - last_selected) / 60 < min_gap_hours:
            continue

        bolus_events = insulin_events[insulin_events["type"] == "bolus"]
        close_boluses = bolus_events[
            np.abs(bolus_events["time_minutes"] - meal_time) < 30
        ]

        if len(close_boluses) == 0:
            continue

        idx_closest = (np.abs(close_boluses["time_minutes"] - meal_time)).argmin()
        bolus = close_boluses.iloc[idx_closest]

        t_start = meal_time - 30
        t_end = meal_time + window_hours * 60

        window_mask = (df["Time_minutes"] >= t_start) & (df["Time_minutes"] <= t_end)
        window_df = df[window_mask].copy()

        if len(window_df) < 10:
            continue

        window_df["Time_minutes_window"] = window_df["Time_minutes"] - t_start

        windows.append(
            {
                "data": window_df,
                "meal_time": meal_time,
                "bolus_dose": bolus["dose_IU"],
                "bolus_time": bolus["time_minutes"],
                "t_start": t_start,
                "t_end": t_end,
                "duration_hours": window_hours,
            }
        )

        last_selected = meal_time

    return windows


def prepare_for_pinn(window_data, Gb_estimate=None, Ib_estimate=10.0):
    """Prepare a meal window for Bayesian PINN training"""
    df = window_data["data"]

    t_minutes = df["Time_minutes_window"].values
    G_obs = df["CGM (mg / dl)"].values

    t_bolus_rel = window_data["bolus_time"] - window_data["t_start"]
    I_obs = insulin_pk_bolus(t_minutes, t_bolus_rel, window_data["bolus_dose"])
    I_obs += Ib_estimate

    if Gb_estimate is None:
        Gb = G_obs[0]
    else:
        Gb = Gb_estimate

    t_mean, t_std = t_minutes.mean(), t_minutes.std()
    G_mean, G_std = G_obs.mean(), G_obs.std()
    I_mean, I_std = I_obs.mean(), I_obs.std()

    if t_std < 1e-6:
        t_std = 1.0
    if G_std < 1e-6:
        G_std = 1.0
    if I_std < 1e-6:
        I_std = 1.0

    return {
        "t_minutes": t_minutes,
        "G_obs": G_obs,
        "I_obs": I_obs,
        "Gb": Gb,
        "Ib": Ib_estimate,
        "normalization": {
            "t_mean": t_mean,
            "t_std": t_std,
            "G_mean": G_mean,
            "G_std": G_std,
            "I_mean": I_mean,
            "I_std": I_std,
        },
        "bolus_dose": window_data["bolus_dose"],
        "meal_time": window_data["meal_time"],
        "window_info": window_data,
    }


def load_patient_data(filepath):
    """Load and prepare patient data"""
    print("=" * 70)
    print("LOADING PATIENT DATA")
    print("=" * 70)

    df = pd.read_excel(filepath, sheet_name=0)
    df["Time_minutes"] = (df["Date"] - df["Date"].min()).dt.total_seconds() / 60

    # Extract insulin events
    insulin_data = []
    for idx, row in df[df["Insulin dose - s.c."].notna()].iterrows():
        dose, insulin_type = extract_insulin_dose(row["Insulin dose - s.c."])
        if dose is not None:
            insulin_data.append(
                {
                    "time_minutes": row["Time_minutes"],
                    "dose_IU": dose,
                    "type": insulin_type,
                    "datetime": row["Date"],
                }
            )

    insulin_events = pd.DataFrame(insulin_data)

    # Compute total insulin
    I_total = compute_total_insulin(df, insulin_events)
    df["Insulin_est"] = I_total

    # Get meal times
    meal_times = df[df["Dietary intake"].notna()]["Time_minutes"].values

    # Extract windows
    print(f"\nExtracting meal-response windows...")
    windows = extract_meal_windows(
        df, insulin_events, meal_times, window_hours=4, min_gap_hours=3
    )

    print(f"Extracted {len(windows)} clean meal windows")

    # Prepare windows for PINN
    prepared_windows = []
    for i, window in enumerate(windows):
        pre_meal_idx = window["data"]["Time_minutes_window"] < 30
        if pre_meal_idx.sum() > 0:
            Gb_est = window["data"][pre_meal_idx]["CGM (mg / dl)"].mean()
        else:
            Gb_est = window["data"]["CGM (mg / dl)"].iloc[0]

        pinn_data = prepare_for_pinn(window, Gb_estimate=Gb_est, Ib_estimate=12.0)
        prepared_windows.append(pinn_data)

    return df, insulin_events, prepared_windows


# ============================================================================
# PART 2: B-PINN MODEL COMPONENTS
# ============================================================================


class PINN(nn.Module):
    """Physics-Informed Neural Network for Bergman model"""

    def __init__(self, hidden_dims=[64, 64, 64], activation="tanh"):
        super().__init__()

        layers = []
        input_dim = 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "softplus":
                layers.append(nn.Softplus())
            else:
                layers.append(nn.Tanh())

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 2))

        self.network = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.network(t)


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

    if fix_p3:
        p3 = torch.tensor(p3_value, dtype=torch.float32)
    else:
        p3 = pyro.sample("p3", dist.LogNormal(torch.tensor(-11.1), torch.tensor(0.5)))

    sigma_G = pyro.sample("sigma_G", dist.HalfNormal(0.5))

    if compute_physics:
        sigma_phys = pyro.sample("sigma_phys", dist.HalfNormal(0.2))

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
            grad_outputs=torch.ones_like(G_hat_norm),
            create_graph=True,
            retain_graph=True,
        )[0]

        dX_dt_norm = torch.autograd.grad(
            X_hat_norm,
            t_phys,
            grad_outputs=torch.ones_like(X_hat_norm),
            create_graph=True,
            retain_graph=True,
        )[0]

        dG_dt = dG_dt_norm * G_std / t_std
        dX_dt = dX_dt_norm / t_std

        R_G = dG_dt + p1 * (G_hat - Gb) + X_hat * G_hat
        R_X = dX_dt + p2 * X_hat - p3 * (I_phys - Ib)

        R_G_norm = R_G / G_std
        R_X_norm = R_X

        with pyro.plate("physics_plate", len(t)):
            pyro.sample(
                "physics_G", dist.Normal(0.0, sigma_phys), obs=R_G_norm.squeeze()
            )
            pyro.sample(
                "physics_X", dist.Normal(0.0, sigma_phys), obs=R_X_norm.squeeze()
            )

        G_pred = G_hat_norm.squeeze()
    else:
        with torch.no_grad():
            out = pinn(t)
            G_pred = out[:, 0]

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

    mean_p1_loc = pyro.param("mean_p1_loc", torch.tensor(-3.58))
    mean_p1_scale = pyro.param(
        "mean_p1_scale", torch.tensor(0.3), constraint=dist.constraints.positive
    )
    pyro.sample("p1", dist.LogNormal(mean_p1_loc, mean_p1_scale))

    mean_p2_loc = pyro.param("mean_p2_loc", torch.tensor(-3.69))
    mean_p2_scale = pyro.param(
        "mean_p2_scale", torch.tensor(0.3), constraint=dist.constraints.positive
    )
    pyro.sample("p2", dist.LogNormal(mean_p2_loc, mean_p2_scale))

    if not fix_p3:
        mean_p3_loc = pyro.param("mean_p3_loc", torch.tensor(-11.1))
        mean_p3_scale = pyro.param(
            "mean_p3_scale", torch.tensor(0.5), constraint=dist.constraints.positive
        )
        pyro.sample("p3", dist.LogNormal(mean_p3_loc, mean_p3_scale))

    sigma_G_scale = pyro.param(
        "sigma_G_scale", torch.tensor(0.5), constraint=dist.constraints.positive
    )
    pyro.sample("sigma_G", dist.HalfNormal(sigma_G_scale))

    if compute_physics:
        sigma_phys_scale = pyro.param(
            "sigma_phys_scale", torch.tensor(0.2), constraint=dist.constraints.positive
        )
        pyro.sample("sigma_phys", dist.HalfNormal(sigma_phys_scale))

    pyro.module("pinn", pinn_net)


def prepare_data_for_training(window_data):
    """Convert window data to tensors and split into train/test"""

    t_minutes = window_data["t_minutes"]
    G_obs = window_data["G_obs"]
    I_obs = window_data["I_obs"]
    norm = window_data["normalization"]

    # Normalize
    t_norm = (t_minutes - norm["t_mean"]) / norm["t_std"]
    G_norm = (G_obs - norm["G_mean"]) / norm["G_std"]
    I_norm = (I_obs - norm["I_mean"]) / norm["I_std"]

    # Convert to tensors
    t = torch.tensor(t_norm, dtype=torch.float32).reshape(-1, 1)
    G = torch.tensor(G_norm, dtype=torch.float32)
    I = torch.tensor(I_norm, dtype=torch.float32).reshape(-1, 1)

    # Split using configuration variable
    n = len(t)
    n_train = int(n * TRAIN_FRACTION)
    indices = torch.randperm(n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = {
        "t": t[train_idx],
        "G_obs": G[train_idx],
        "I_obs": I[train_idx],
        "Gb": window_data["Gb"],
        "Ib": window_data["Ib"],
        "normalization": norm,
    }

    test_data = {
        "t": t[test_idx],
        "G_obs": G[test_idx],
        "I_obs": I[test_idx],
        "Gb": window_data["Gb"],
        "Ib": window_data["Ib"],
        "normalization": norm,
    }

    full_data = {
        "t": t,
        "G_obs": G,
        "I_obs": I,
        "t_raw": torch.tensor(t_minutes, dtype=torch.float32),
        "G_obs_raw": torch.tensor(G_obs, dtype=torch.float32),
        "I_obs_raw": torch.tensor(I_obs, dtype=torch.float32),
        "Gb": window_data["Gb"],
        "Ib": window_data["Ib"],
        "normalization": norm,
        "window_info": window_data.get("window_info", {}),
    }

    return train_data, test_data, full_data


def train_bayesian_pinn(train_data, val_data, save_dir="./results"):
    """Train Bayesian PINN"""

    print("\n" + "=" * 70)
    print("TRAINING BAYESIAN PINN")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)

    pyro.clear_param_store()

    pinn_net = PINN(hidden_dims=NETWORK_HIDDEN_DIMS)

    optimizer = ClippedAdam({"lr": LEARNING_RATE, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    t = train_data["t"]
    G_obs = train_data["G_obs"]
    I_obs = train_data["I_obs"]
    Gb = train_data["Gb"]
    Ib = train_data["Ib"]
    norm_params = train_data["normalization"]

    t_val = val_data["t"]
    G_obs_val = val_data["G_obs"]
    I_obs_val = val_data["I_obs"]

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nConfiguration:")
    print(f"  Training samples: {len(t)}")
    print(f"  Validation samples: {len(t_val)}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Fix p3: {FIX_P3}")
    print(f"  Network architecture: {NETWORK_HIDDEN_DIMS}")

    for iteration in range(ITERATIONS):
        loss = svi.step(
            t, G_obs, I_obs, Gb, Ib, norm_params, pinn_net, FIX_P3, 1e-5, True
        )
        train_losses.append(loss)

        if (iteration + 1) % 50 == 0:
            val_loss = svi.evaluate_loss(
                t_val,
                G_obs_val,
                I_obs_val,
                Gb,
                Ib,
                norm_params,
                pinn_net,
                FIX_P3,
                1e-5,
                False,
            )
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                best_checkpoint = {
                    "iteration": iteration + 1,
                    "train_loss": loss,
                    "val_loss": val_loss,
                    "pinn_state_dict": pinn_net.state_dict(),
                    "pyro_param_store": pyro.get_param_store().get_state(),
                }
                torch.save(best_checkpoint, os.path.join(save_dir, "best_model.pt"))
            else:
                patience_counter += 50

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping at iteration {iteration + 1}")
                    break

        if (iteration + 1) % 500 == 0 or iteration == 0:
            msg = f"Iter {iteration + 1:5d} | Train Loss: {loss:.2e}"
            if len(val_losses) > 0:
                msg += f" | Val Loss: {val_losses[-1]:.2e}"
            print(msg)

    print("\nTraining completed!")
    return pinn_net, train_losses, val_losses


# ============================================================================
# PART 3: ANALYSIS AND VISUALIZATION
# ============================================================================


def analyze_posterior(pinn_net, full_data):
    """Sample from posterior and compute statistics"""

    print("\n" + "=" * 70)
    print("POSTERIOR ANALYSIS")
    print("=" * 70)

    predictive = Predictive(model, guide=guide, num_samples=POSTERIOR_SAMPLES)

    samples = predictive(
        full_data["t"],
        full_data["G_obs"],
        full_data["I_obs"],
        full_data["Gb"],
        full_data["Ib"],
        full_data["normalization"],
        pinn_net,
        FIX_P3,
        1e-5,
        False,
    )

    p1_samples = samples["p1"].detach().numpy()
    p2_samples = samples["p2"].detach().numpy()

    print("\nParameter Posterior Statistics:")
    print(f"{'Parameter':<10} {'Mean':<12} {'Std':<12} {'2.5%':<12} {'97.5%':<12}")
    print("-" * 60)

    print(
        f"{'p1':<10} {p1_samples.mean():<12.6f} {p1_samples.std():<12.6f} "
        f"{np.percentile(p1_samples, 2.5):<12.6f} {np.percentile(p1_samples, 97.5):<12.6f}"
    )

    print(
        f"{'p2':<10} {p2_samples.mean():<12.6f} {p2_samples.std():<12.6f} "
        f"{np.percentile(p2_samples, 2.5):<12.6f} {np.percentile(p2_samples, 97.5):<12.6f}"
    )

    if not FIX_P3 and "p3" in samples:
        p3_samples = samples["p3"].detach().numpy()
        print(
            f"{'p3':<10} {p3_samples.mean():<12.6e} {p3_samples.std():<12.6e} "
            f"{np.percentile(p3_samples, 2.5):<12.6e} {np.percentile(p3_samples, 97.5):<12.6e}"
        )

    return samples


def predict_with_uncertainty(pinn_net, full_data, samples, n_posterior_samples=100):
    """Generate predictions with uncertainty"""

    n_available = len(samples["p1"])
    indices = np.random.choice(
        n_available, min(n_posterior_samples, n_available), replace=False
    )

    predictions_G = []

    with torch.no_grad():
        for idx in indices:
            out = pinn_net(full_data["t"])
            predictions_G.append(out[:, 0].numpy())

    predictions_G = np.array(predictions_G)

    norm = full_data["normalization"]
    predictions_G = predictions_G * norm["G_std"] + norm["G_mean"]

    return {
        "G_mean": predictions_G.mean(axis=0),
        "G_std": predictions_G.std(axis=0),
        "G_lower": np.percentile(predictions_G, 2.5, axis=0),
        "G_upper": np.percentile(predictions_G, 97.5, axis=0),
    }


def compute_metrics(full_data, predictions):
    """Compute performance metrics"""

    G_true = full_data["G_obs_raw"].numpy()
    G_pred = predictions["G_mean"]

    rmse = np.sqrt(np.mean((G_true - G_pred) ** 2))
    mae = np.mean(np.abs(G_true - G_pred))

    ss_res = np.sum((G_true - G_pred) ** 2)
    ss_tot = np.sum((G_true - G_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    in_ci = (G_true >= predictions["G_lower"]) & (G_true <= predictions["G_upper"])
    coverage = in_ci.mean() * 100

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"RMSE:            {rmse:.2f} mg/dL")
    print(f"MAE:             {mae:.2f} mg/dL")
    print(f"R²:              {r2:.4f}")
    print(f"95% CI Coverage: {coverage:.1f}%")

    return {"rmse": rmse, "mae": mae, "r2": r2, "coverage": coverage}


def plot_comprehensive_results(
    pinn_net, full_data, train_losses, val_losses, samples, predictions, save_dir
):
    """Create comprehensive visualization"""

    fig = plt.figure(figsize=(18, 12))

    t_raw = full_data["t_raw"].numpy().flatten()
    G_obs_raw = full_data["G_obs_raw"].numpy()

    # Plot 1: Training curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label="Train", alpha=0.7)
    if val_losses:
        val_iters = np.linspace(0, len(train_losses), len(val_losses))
        ax1.plot(val_iters, val_losses, label="Validation", alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ELBO Loss")
    ax1.set_title("Training Progress")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Glucose predictions
    ax2 = plt.subplot(3, 3, 2)
    ax2.fill_between(
        t_raw, predictions["G_lower"], predictions["G_upper"], alpha=0.3, label="95% CI"
    )
    ax2.plot(t_raw, G_obs_raw, "o", alpha=0.5, markersize=4, label="Observed")
    ax2.plot(t_raw, predictions["G_mean"], "-", linewidth=2, label="Predicted")
    ax2.axhline(
        y=full_data["Gb"], color="r", linestyle="--", alpha=0.5, label="Baseline"
    )
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Glucose (mg/dL)")
    ax2.set_title("Glucose Predictions with Uncertainty")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals
    ax3 = plt.subplot(3, 3, 3)
    residuals = G_obs_raw - predictions["G_mean"]
    ax3.scatter(t_raw, residuals, alpha=0.5, s=10)
    ax3.axhline(y=0, color="r", linestyle="--")
    ax3.fill_between(
        t_raw,
        -2 * predictions["G_std"],
        2 * predictions["G_std"],
        alpha=0.2,
        color="gray",
    )
    ax3.set_xlabel("Time (minutes)")
    ax3.set_ylabel("Residual (mg/dL)")
    ax3.set_title("Prediction Residuals")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Insulin input
    ax4 = plt.subplot(3, 3, 4)
    I_raw = full_data["I_obs_raw"].numpy().flatten()
    ax4.plot(t_raw, I_raw, linewidth=2)
    ax4.axhline(
        y=full_data["Ib"],
        color="r",
        linestyle="--",
        label=f"Ib = {full_data['Ib']:.1f}",
    )

    # Mark bolus if available
    if "window_info" in full_data and "bolus_time" in full_data["window_info"]:
        bolus_time = (
            full_data["window_info"]["bolus_time"] - full_data["window_info"]["t_start"]
        )
        ax4.axvline(
            x=bolus_time,
            color="green",
            linestyle=":",
            alpha=0.7,
            label=f"Bolus {full_data['window_info']['bolus_dose']:.0f} IU",
        )

    ax4.set_xlabel("Time (minutes)")
    ax4.set_ylabel("Insulin (μU/mL)")
    ax4.set_title("Estimated Insulin Concentration")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Parameter posterior - p1
    ax5 = plt.subplot(3, 3, 5)
    p1_samples = samples["p1"].detach().numpy()
    ax5.hist(p1_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax5.axvline(
        p1_samples.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {p1_samples.mean():.4f}",
        linewidth=2,
    )
    ax5.set_xlabel("p1 (min⁻¹)")
    ax5.set_ylabel("Density")
    ax5.set_title("Posterior: Glucose Effectiveness")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Parameter posterior - p2
    ax6 = plt.subplot(3, 3, 6)
    p2_samples = samples["p2"].detach().numpy()
    ax6.hist(p2_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
    ax6.axvline(
        p2_samples.mean(),
        color="r",
        linestyle="--",
        label=f"Mean: {p2_samples.mean():.4f}",
        linewidth=2,
    )
    ax6.set_xlabel("p2 (min⁻¹)")
    ax6.set_ylabel("Density")
    ax6.set_title("Posterior: Insulin Action Decay")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Parameter posterior - p3 or correlation
    ax7 = plt.subplot(3, 3, 7)
    if "p3" in samples:
        p3_samples = samples["p3"].detach().numpy()
        ax7.hist(p3_samples, bins=50, alpha=0.7, edgecolor="black", density=True)
        ax7.axvline(
            p3_samples.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {p3_samples.mean():.6f}",
            linewidth=2,
        )
        ax7.set_xlabel("p3 (min⁻² per μU/mL)")
        ax7.set_ylabel("Density")
        ax7.set_title("Posterior: Insulin Sensitivity")
        ax7.legend()
    else:
        ax7.scatter(p1_samples, p2_samples, alpha=0.3, s=10)
        ax7.set_xlabel("p1")
        ax7.set_ylabel("p2")
        ax7.set_title("Parameter Correlation: p1 vs p2")
        corr = np.corrcoef(p1_samples, p2_samples)[0, 1]
        ax7.text(
            0.05,
            0.95,
            f"ρ = {corr:.3f}",
            transform=ax7.transAxes,
            verticalalignment="top",
        )
    ax7.grid(True, alpha=0.3)

    # Plot 8: Prediction intervals over time
    ax8 = plt.subplot(3, 3, 8)
    uncertainty_width = predictions["G_upper"] - predictions["G_lower"]
    ax8.plot(t_raw, uncertainty_width, linewidth=2, color="purple")
    ax8.set_xlabel("Time (minutes)")
    ax8.set_ylabel("95% CI Width (mg/dL)")
    ax8.set_title("Prediction Uncertainty Over Time")
    ax8.grid(True, alpha=0.3)

    # Plot 9: Q-Q plot
    ax9 = plt.subplot(3, 3, 9)
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=ax9)
    ax9.set_title("Q-Q Plot: Residual Normality")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "comprehensive_results.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"✓ Comprehensive results saved to: {save_dir}/comprehensive_results.png")
    plt.close()


# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================


def run_single_window_analysis(window_idx, prepared_windows, base_results_dir):
    """Run complete analysis on a single window"""

    print("\n" + "=" * 70)
    print(f"ANALYZING WINDOW {window_idx + 1}")
    print("=" * 70)

    window_data = prepared_windows[window_idx]

    # Create results directory for this window
    window_dir = os.path.join(base_results_dir, f"window_{window_idx}")
    os.makedirs(window_dir, exist_ok=True)

    # Print window info
    print(f"\nWindow Information:")
    print(f"  Meal time: {window_data['meal_time']/60:.1f} hours")
    print(f"  Bolus dose: {window_data['bolus_dose']:.1f} IU")
    print(f"  Baseline glucose: {window_data['Gb']:.1f} mg/dL")
    print(f"  Baseline insulin: {window_data['Ib']:.1f} μU/mL")
    print(f"  Number of samples: {len(window_data['t_minutes'])}")

    # Prepare data
    train_data, test_data, full_data = prepare_data_for_training(window_data)

    # Train model
    pinn_net, train_losses, val_losses = train_bayesian_pinn(
        train_data, test_data, save_dir=window_dir
    )

    # Analyze posterior
    samples = analyze_posterior(pinn_net, full_data)

    # Generate predictions
    predictions = predict_with_uncertainty(
        pinn_net, full_data, samples, n_posterior_samples=100
    )

    # Compute metrics
    metrics = compute_metrics(full_data, predictions)

    # Create visualizations
    plot_comprehensive_results(
        pinn_net, full_data, train_losses, val_losses, samples, predictions, window_dir
    )

    # Save results
    results = {
        "window_idx": window_idx,
        "window_info": window_data.get("window_info", {}),
        "parameters": {
            "p1": {
                "mean": samples["p1"].mean().item(),
                "std": samples["p1"].std().item(),
                "ci_lower": np.percentile(samples["p1"].numpy(), 2.5),
                "ci_upper": np.percentile(samples["p1"].numpy(), 97.5),
            },
            "p2": {
                "mean": samples["p2"].mean().item(),
                "std": samples["p2"].std().item(),
                "ci_lower": np.percentile(samples["p2"].numpy(), 2.5),
                "ci_upper": np.percentile(samples["p2"].numpy(), 97.5),
            },
        },
        "metrics": metrics,
        "fix_p3": FIX_P3,
    }

    if not FIX_P3 and "p3" in samples:
        results["parameters"]["p3"] = {
            "mean": samples["p3"].mean().item(),
            "std": samples["p3"].std().item(),
            "ci_lower": np.percentile(samples["p3"].numpy(), 2.5),
            "ci_upper": np.percentile(samples["p3"].numpy(), 97.5),
        }

    # Save to pickle
    with open(os.path.join(window_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # Save to text file
    with open(os.path.join(window_dir, "summary.txt"), "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"WINDOW {window_idx + 1} ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("Window Information:\n")
        f.write(f"  Meal time: {window_data['meal_time']/60:.1f} hours\n")
        f.write(f"  Bolus dose: {window_data['bolus_dose']:.1f} IU\n")
        f.write(f"  Baseline glucose: {window_data['Gb']:.1f} mg/dL\n\n")

        f.write("Estimated Parameters:\n")
        for param_name, param_data in results["parameters"].items():
            f.write(f"  {param_name}: {param_data['mean']:.6e} ")
            f.write(
                f"(95% CI: [{param_data['ci_lower']:.6e}, {param_data['ci_upper']:.6e}])\n"
            )

        f.write("\nPerformance Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"  {metric_name}: {metric_value:.4f}\n")

    print(f"\n✓ Results saved to: {window_dir}")

    return results


def main():
    """Main execution function"""

    print("=" * 70)
    print("BAYESIAN PINN ANALYSIS FOR PATIENT GLUCOSE DATA")
    print("=" * 70)

    # Load data
    df, insulin_events, prepared_windows = load_patient_data(DATA_FILE)

    print(f"\nTotal windows available: {len(prepared_windows)}")

    # Create base results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if RUN_ALL_WINDOWS:
        # Run on all available windows
        print("\n" + "=" * 70)
        print("RUNNING ANALYSIS ON ALL WINDOWS")
        print("=" * 70)

        # Analyze all windows
        window_indices = list(range(len(prepared_windows)))
        print(f"\nAnalyzing all {len(window_indices)} windows")

        all_results = []
        failed_windows = []

        for idx in window_indices:
            try:
                result = run_single_window_analysis(idx, prepared_windows, RESULTS_DIR)
                all_results.append(result)
            except Exception as e:
                print(f"\n⚠ Warning: Window {idx} failed with error: {str(e)}")
                print("Continuing with next window...")
                failed_windows.append({"window_idx": idx, "error": str(e)})
                continue

        # Save combined results
        with open(os.path.join(RESULTS_DIR, "all_results.pkl"), "wb") as f:
            pickle.dump(all_results, f)

        # Create summary report
        with open(os.path.join(RESULTS_DIR, "all_windows_summary.txt"), "w") as f:
            f.write("=" * 70 + "\n")
            f.write("ALL WINDOWS ANALYSIS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total windows processed: {len(all_results)}\n")
            f.write(f"Failed windows: {len(failed_windows)}\n\n")

            if failed_windows:
                f.write("Failed Windows:\n")
                for fw in failed_windows:
                    f.write(f"  Window {fw['window_idx']}: {fw['error']}\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("WINDOW-BY-WINDOW SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            for result in all_results:
                f.write(f"Window {result['window_idx'] + 1}:\n")
                f.write(f"  p1: {result['parameters']['p1']['mean']:.6f}\n")
                f.write(f"  p2: {result['parameters']['p2']['mean']:.6f}\n")
                if "p3" in result["parameters"]:
                    f.write(f"  p3: {result['parameters']['p3']['mean']:.6e}\n")
                f.write(f"  RMSE: {result['metrics']['rmse']:.2f} mg/dL\n")
                f.write(f"  R²: {result['metrics']['r2']:.4f}\n\n")

        print("\n" + "=" * 70)
        print("MULTI-WINDOW ANALYSIS COMPLETE")
        print("=" * 70)
        print(
            f"Successfully analyzed: {len(all_results)}/{len(window_indices)} windows"
        )
        print(f"Failed windows: {len(failed_windows)}")
        print(f"Results saved to: {RESULTS_DIR}")

    else:
        # Run on single window
        if WINDOW >= len(prepared_windows):
            print(f"\nError: Window {WINDOW} does not exist.")
            print(f"Available windows: 0-{len(prepared_windows)-1}")
            return

        result = run_single_window_analysis(WINDOW, prepared_windows, RESULTS_DIR)

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nResults for Window {WINDOW}:")
        print(f"  p1 (glucose effectiveness): {result['parameters']['p1']['mean']:.6f}")
        print(f"  p2 (insulin action decay):  {result['parameters']['p2']['mean']:.6f}")
        if "p3" in result["parameters"]:
            print(
                f"  p3 (insulin sensitivity):   {result['parameters']['p3']['mean']:.6e}"
            )
        print(f"\n  RMSE: {result['metrics']['rmse']:.2f} mg/dL")
        print(f"  R²:   {result['metrics']['r2']:.4f}")
        print(f"  95% CI Coverage: {result['metrics']['coverage']:.1f}%")


if __name__ == "__main__":
    main()