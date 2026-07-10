"""
Prepare Real Patient Data for Bayesian PINN Analysis
=====================================================
Processes CGM data and prepares meal windows for B-PINN training.
Plots are styled for scientific publication (300 dpi PDF, booktabs-compatible).
"""

import re
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ── Global publication style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Times New Roman", "DejaVu Serif"],
    "font.size"          : 10,
    "axes.titlesize"     : 11,
    "axes.labelsize"     : 10,
    "xtick.labelsize"    : 9,
    "ytick.labelsize"    : 9,
    "legend.fontsize"    : 9,
    "figure.dpi"         : 300,
    "axes.linewidth"     : 0.8,
    "xtick.direction"    : "in",
    "ytick.direction"    : "in",
    "xtick.major.size"   : 3.5,
    "ytick.major.size"   : 3.5,
    "xtick.minor.size"   : 2.0,
    "ytick.minor.size"   : 2.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth"    : 1.4,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.05,
    "pdf.fonttype"       : 42,
    "ps.fonttype"        : 42,
})

# Colour palette — colourblind-safe (Wong 2011)
C = {
    "cgm"      : "#0072B2",   # blue
    "cbg"      : "#D55E00",   # vermillion
    "target"   : "#E69F00",   # amber
    "hypo"     : "#CC79A7",   # reddish-purple
    "bolus"    : "#0072B2",   # blue
    "basal"    : "#E69F00",   # amber
    "meal"     : "#009E73",   # green
    "baseline" : "#CC79A7",   # purple
    "grid"     : "#cccccc",
}


# ============================================================================
# PHARMACOKINETIC MODELS
# ============================================================================

def extract_insulin_dose(text):
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
    scaling_factor = 15.0
    dt = t - t_dose
    I = np.zeros_like(dt)
    mask = dt >= 0
    tau_abs  = tau
    tau_elim = tau * 3
    I[mask] = (scaling_factor * dose_IU * bioavail *
               (np.exp(-dt[mask] / tau_elim) - np.exp(-dt[mask] / tau_abs)))
    peak = I.max()
    if peak > 0:
        expected_peak = scaling_factor * dose_IU * bioavail * 0.4
        I = I * (expected_peak / peak)
    return I


def insulin_pk_basal(t, t_dose, dose_IU, tau=600, steady_state_conc=None):
    scaling_factor = 8.0
    dt = t - t_dose
    I  = np.zeros_like(dt)
    mask = dt >= 0
    plateau = scaling_factor * dose_IU * 0.7
    I[mask] = plateau * (1 - np.exp(-dt[mask] / tau))
    return I


def compute_total_insulin(df, insulin_events):
    t       = df["Time_minutes"].values
    I_total = np.zeros_like(t)
    basal_events = insulin_events[insulin_events["type"] == "basal"]
    if len(basal_events) > 0:
        I_total += basal_events["dose_IU"].mean() * 5.0
    for _, event in insulin_events[insulin_events["type"] == "bolus"].iterrows():
        I_total += insulin_pk_bolus(t, event["time_minutes"], event["dose_IU"])
    for _, event in basal_events.iterrows():
        I_total += insulin_pk_basal(t, event["time_minutes"], event["dose_IU"])
    return I_total


def extract_meal_windows(df, insulin_events, meal_times,
                         window_hours=4, min_gap_hours=2):
    windows       = []
    last_selected = -np.inf
    for meal_time in meal_times:
        if (meal_time - last_selected) / 60 < min_gap_hours:
            continue
        bolus_events  = insulin_events[insulin_events["type"] == "bolus"]
        close_boluses = bolus_events[
            np.abs(bolus_events["time_minutes"] - meal_time) < 30]
        if len(close_boluses) == 0:
            continue
        idx_closest = (np.abs(close_boluses["time_minutes"] - meal_time)).argmin()
        bolus   = close_boluses.iloc[idx_closest]
        t_start = meal_time - 30
        t_end   = meal_time + window_hours * 60
        mask    = (df["Time_minutes"] >= t_start) & (df["Time_minutes"] <= t_end)
        wdf     = df[mask].copy()
        if len(wdf) < 10:
            continue
        wdf["Time_minutes_window"] = wdf["Time_minutes"] - t_start
        windows.append({
            "data"          : wdf,
            "meal_time"     : meal_time,
            "bolus_dose"    : bolus["dose_IU"],
            "bolus_time"    : bolus["time_minutes"],
            "t_start"       : t_start,
            "t_end"         : t_end,
            "duration_hours": window_hours,
        })
        last_selected = meal_time
    return windows


def prepare_for_pinn(window_data, Gb_estimate=None, Ib_estimate=10.0):
    df        = window_data["data"]
    t_minutes = df["Time_minutes_window"].values
    G_obs     = df["CGM (mg / dl)"].values
    t_bolus_rel = window_data["bolus_time"] - window_data["t_start"]
    I_obs     = insulin_pk_bolus(t_minutes, t_bolus_rel, window_data["bolus_dose"])
    I_obs    += Ib_estimate
    Gb        = G_obs[0] if Gb_estimate is None else Gb_estimate
    t_mean, t_std = t_minutes.mean(), max(t_minutes.std(), 1e-6)
    G_mean, G_std = G_obs.mean(),     max(G_obs.std(),     1e-6)
    I_mean, I_std = I_obs.mean(),     max(I_obs.std(),     1e-6)
    return {
        "t_minutes"    : t_minutes,
        "G_obs"        : G_obs,
        "I_obs"        : I_obs,
        "Gb"           : Gb,
        "Ib"           : Ib_estimate,
        "normalization": dict(t_mean=t_mean, t_std=t_std,
                              G_mean=G_mean, G_std=G_std,
                              I_mean=I_mean, I_std=I_std),
        "bolus_dose"   : window_data["bolus_dose"],
        "meal_time"    : window_data["meal_time"],
        "window_info"  : window_data,
    }


# ============================================================================
# PUBLICATION-QUALITY PLOTS
# ============================================================================

def plot_patient_overview(df, insulin_events, save_path):
    """
    Three-panel overview: CGM trace | insulin doses | meal events.
    Saved as PDF at 300 dpi with embedded fonts.
    """
    fig = plt.figure(figsize=(8.5, 7.0))

    # Proportional panel heights: glucose gets most space
    gs = fig.add_gridspec(3, 1, hspace=0.08,
                          height_ratios=[3, 1.6, 0.9],
                          left=0.10, right=0.97,
                          top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    t_h = df["Time_minutes"] / 60   # convert to hours

    # ── Panel 1: CGM ────────────────────────────────────────────────────────
    ax1.plot(t_h, df["CGM (mg / dl)"],
             color=C["cgm"], lw=1.2, alpha=0.9, label="CGM", zorder=3)

    cbg_mask = df["CBG (mg / dl)"].notna()
    if cbg_mask.sum() > 0:
        ax1.scatter(df[cbg_mask]["Time_minutes"] / 60,
                    df[cbg_mask]["CBG (mg / dl)"],
                    color=C["cbg"], s=18, zorder=5, alpha=0.85,
                    edgecolors="white", linewidths=0.4,
                    label="CBG (fingerstick)")

    ax1.axhline(180, color=C["target"], lw=0.9, ls="--", alpha=0.8,
                label="Upper target (180 mg·dL$^{-1}$)")
    ax1.axhline(70,  color=C["hypo"],   lw=0.9, ls=":",  alpha=0.8,
                label="Hypoglycaemia (70 mg·dL$^{-1}$)")

    # Shade hypoglycaemic zone
    ax1.axhspan(0, 70, color=C["hypo"], alpha=0.06, lw=0)

    ax1.set_ylabel("Glucose (mg·dL$^{-1}$)")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", framealpha=0.9,
               handlelength=1.8, handletextpad=0.5)
    ax1.grid(True, color=C["grid"], lw=0.5, ls="--", alpha=0.7)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: Insulin doses ───────────────────────────────────────────────
    bolus_df = insulin_events[insulin_events["type"] == "bolus"]
    basal_df = insulin_events[insulin_events["type"] == "basal"]

    markerline, stemlines, baseline = ax2.stem(
        bolus_df["time_minutes"] / 60, bolus_df["dose_IU"],
        linefmt=C["bolus"], markerfmt="o", basefmt=" ")
    plt.setp(stemlines,    lw=1.0, alpha=0.8)
    plt.setp(markerline,   ms=4,   color=C["bolus"], alpha=0.9,
             markeredgecolor="white", markeredgewidth=0.4)

    markerline2, stemlines2, _ = ax2.stem(
        basal_df["time_minutes"] / 60, basal_df["dose_IU"],
        linefmt=C["basal"], markerfmt="s", basefmt=" ")
    plt.setp(stemlines2,   lw=1.0, alpha=0.8, color=C["basal"])
    plt.setp(markerline2,  ms=4,   color=C["basal"], alpha=0.9,
             markeredgecolor="white", markeredgewidth=0.4)

    legend_elems = [
        Line2D([0],[0], color=C["bolus"], lw=1.2,
               marker="o", ms=5, label="Bolus (rapid-acting)"),
        Line2D([0],[0], color=C["basal"], lw=1.2,
               marker="s", ms=5, label="Basal (long-acting)"),
    ]
    ax2.legend(handles=legend_elems, loc="upper right", framealpha=0.9,
               handlelength=1.8)
    ax2.set_ylabel("Dose (IU)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, color=C["grid"], lw=0.5, ls="--", alpha=0.7)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3: Meal events ─────────────────────────────────────────────────
    meal_t = df[df["Dietary intake"].notna()]["Time_minutes"] / 60
    ax3.vlines(meal_t, 0, 1,
               color=C["meal"], lw=1.2, alpha=0.7,
               transform=ax3.get_xaxis_transform())
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_ylabel("Meals", labelpad=14)
    ax3.set_xlabel("Time (h)")
    ax3.grid(True, color=C["grid"], lw=0.5, ls="--", alpha=0.7, axis="x")

    # shared x-axis limits
    x_max = df["Time_minutes"].max() / 60
    ax1.set_xlim(0, x_max)
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    plt.savefig(save_path, dpi=300)
    print(f"✓ Patient overview  → {save_path}")
    plt.close()


def plot_meal_windows(prepared_windows, save_path, n_cols=3):
    """
    Grid of individual meal-response windows.
    Each panel: CGM trajectory, pre-meal baseline, bolus marker.
    """
    n_plot = min(6, len(prepared_windows))
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.35,
                        left=0.10, right=0.97,
                        top=0.91, bottom=0.10)

    for i, pdata in enumerate(prepared_windows[:n_plot]):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        t   = pdata["t_minutes"]
        G   = pdata["G_obs"]
        Gb  = pdata["Gb"]
        bolus_t = (pdata["window_info"]["bolus_time"]
                   - pdata["window_info"]["t_start"])

        # Glucose trace
        ax.plot(t, G, color=C["cgm"], lw=1.2, alpha=0.9,
                marker="o", ms=2.5, markeredgewidth=0,
                zorder=3, label="CGM")

        # Pre-meal baseline
        ax.axhline(Gb, color=C["baseline"], lw=0.9, ls="--", alpha=0.8,
                   label=f"$G_b$ = {Gb:.0f}")

        # Bolus timing
        ax.axvline(bolus_t, color=C["meal"], lw=1.0, ls=":",
                   alpha=0.85,
                   label=f"Bolus {pdata['bolus_dose']:.0f} IU")

        # Shade post-bolus region lightly
        ax.axvspan(bolus_t, t.max(), color=C["meal"], alpha=0.04, lw=0)

        ax.set_title(f"Window {i + 1}", fontsize=9, pad=3)
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.set_ylabel("Glucose\n(mg·dL$^{-1}$)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.85, handlelength=1.4,
                  handletextpad=0.4, borderpad=0.4)
        ax.grid(True, color=C["grid"], lw=0.4, ls="--", alpha=0.7)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    # Hide unused panels
    for i in range(n_plot, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    plt.savefig(save_path, dpi=300)
    print(f"✓ Meal windows      → {save_path}")
    plt.close()


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_statistics(df, insulin_events):
    print("\n" + "=" * 70)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\n--- GLUCOSE DATA ---")
    print(f"Total CGM readings    : {len(df)}")
    print(f"Duration              : {df['Time_minutes'].max() / (60*24):.1f} days")
    print(f"Sampling interval     : ~{df['Time_minutes'].diff().median():.0f} min")
    print(f"Glucose range         : {df['CGM (mg / dl)'].min():.1f} – "
          f"{df['CGM (mg / dl)'].max():.1f} mg/dL")
    print(f"Glucose mean ± SD     : {df['CGM (mg / dl)'].mean():.1f} ± "
          f"{df['CGM (mg / dl)'].std():.1f} mg/dL")
    cbg_count = df["CBG (mg / dl)"].notna().sum()
    if cbg_count > 0:
        print(f"CBG calibrations      : {cbg_count}")

    print(f"\n--- INSULIN ---")
    for itype in ["bolus", "basal"]:
        ev = insulin_events[insulin_events["type"] == itype]
        print(f"{itype.capitalize():6s}: {len(ev)} events  |  "
              f"mean dose {ev['dose_IU'].mean():.1f} ± "
              f"{ev['dose_IU'].std():.1f} IU  |  "
              f"range {ev['dose_IU'].min():.0f}–{ev['dose_IU'].max():.0f} IU")

    print(f"\n--- MEALS ---")
    print(f"Total meal events     : {df['Dietary intake'].notna().sum()}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PATIENT DATA PREPARATION FOR BAYESIAN PINN")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(DATA_FILE, sheet_name=0)
    df["Time_minutes"] = (df["Date"] - df["Date"].min()).dt.total_seconds() / 60

    insulin_data = []
    for idx, row in df[df["Insulin dose - s.c."].notna()].iterrows():
        dose, insulin_type = extract_insulin_dose(row["Insulin dose - s.c."])
        if dose is not None:
            insulin_data.append({
                "time_minutes": row["Time_minutes"],
                "dose_IU"     : dose,
                "type"        : insulin_type,
                "datetime"    : row["Date"],
            })
    insulin_events = pd.DataFrame(insulin_data)

    print_summary_statistics(df, insulin_events)

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    plot_patient_overview(
        df, insulin_events,
        save_path=os.path.join(OUTPUT_DIR, "patient_data_overview.pdf")
    )

    I_total = compute_total_insulin(df, insulin_events)
    df["Insulin_est"] = I_total

    meal_times = df[df["Dietary intake"].notna()]["Time_minutes"].values
    print(f"\nExtracting meal-response windows (total meals logged: {len(meal_times)})...")
    windows = extract_meal_windows(
        df, insulin_events, meal_times, window_hours=4, min_gap_hours=3)
    print(f"Extracted {len(windows)} clean meal windows.")

    print("\n" + "=" * 70)
    print("PREPARING WINDOWS FOR PINN TRAINING")
    print("=" * 70)
    prepared_windows = []
    for i, window in enumerate(windows):
        pre_meal = window["data"]["Time_minutes_window"] < 30
        Gb_est   = (window["data"][pre_meal]["CGM (mg / dl)"].mean()
                    if pre_meal.sum() > 0
                    else window["data"]["CGM (mg / dl)"].iloc[0])
        pdata = prepare_for_pinn(window, Gb_estimate=Gb_est, Ib_estimate=12.0)
        prepared_windows.append(pdata)
        print(f"  Window {i+1:>2d}: t={window['meal_time']/60:.1f} h  |  "
              f"Bolus {window['bolus_dose']:.0f} IU  |  "
              f"Gb={pdata['Gb']:.1f} mg/dL  |  "
              f"n={len(pdata['t_minutes'])} pts")

    plot_meal_windows(
        prepared_windows,
        save_path=os.path.join(OUTPUT_DIR, "meal_windows_prepared.pdf")
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Total windows prepared : {len(prepared_windows)}")

    return df, insulin_events, prepared_windows


if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION — edit these
    # ============================================================================
    DATA_FILE  = "/content/drive/MyDrive/Colab Notebooks/bpinns-results/shangai/1006_1_20210209/1006_1_20210209.xlsx"
    OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/bpinns-results/shangai/1006_1_20210209"
    # ============================================================================

    df, insulin_events, windows = main()

    print("\n✓ Data prepared and ready for Bayesian PINN analysis")

    try:
        import pickle
        with open(os.path.join(OUTPUT_DIR, "patient_windows.pkl"), "wb") as f:
            pickle.dump(windows, f)
        print(f"✓ Windows saved to: {os.path.join(OUTPUT_DIR, 'patient_windows.pkl')}")
    except Exception as e:
        print(f"Note: Could not save windows pickle: {e}")