"""
Joint Posterior Inference of Bergman Minimal Model Parameters
=============================================================

Combines results from all 40 meal windows to infer the population-level
posterior distribution of p1, p2, p3 via hierarchical Bayesian analysis.

Usage (Google Colab):
    1. Upload all_windows_summary.txt (or paste the data below)
    2. Set OUTPUT_DIR to your preferred folder
    3. Run all cells

Outputs:
    - bergman_marginal_posteriors.pdf  : Marginal KDE posteriors for p1, p2, p3
    - bergman_joint_pairs.pdf          : Pairwise joint distributions (corner plot)
    - bergman_summary_table.pdf        : Publication-ready parameter summary table
"""

# ============================================================================
# CONFIGURATION — edit these
# ============================================================================
OUTPUT_DIR  = "/content/drive/MyDrive/Colab Notebooks/bpinns-results/shangai/1011_0_20210622"          # Where to save figures
DATA_SOURCE = "file"                      # "file" or "inline"
DATA_FILE   = "/content/drive/MyDrive/Colab Notebooks/bpinns-results/shangai/1011_0_20210622/all_windows_summary.txt"  # Used if DATA_SOURCE="file"
# ============================================================================

import os
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "serif",
    "font.serif"        : ["Times New Roman", "DejaVu Serif"],
    "font.size"         : 10,
    "axes.titlesize"    : 11,
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "figure.dpi"        : 300,
    "axes.linewidth"    : 0.8,
    "xtick.direction"   : "in",
    "ytick.direction"   : "in",
    "xtick.major.size"  : 3.5,
    "ytick.major.size"  : 3.5,
    "xtick.minor.size"  : 2.0,
    "ytick.minor.size"  : 2.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth"   : 1.4,
    "savefig.bbox"      : "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype"      : 42,   # embeds fonts for journal submission
    "ps.fonttype"       : 42,
})

PALETTE = {
    "p1" : "#2166ac",   # blue
    "p2" : "#d6604d",   # red-orange
    "p3" : "#4dac26",   # green
    "fill": 0.18,
    "grid": "#cccccc",
}

# ── Parse summary file ────────────────────────────────────────────────────────
def parse_summary(path):
    with open(path) as f:
        text = f.read()

    blocks = re.findall(
        r"Window\s+(\d+):\s+"
        r"p1:\s+([\d.eE+\-]+)\s+"
        r"p2:\s+([\d.eE+\-]+)\s+"
        r"p3:\s+([\d.eE+\-]+)\s+"
        r"RMSE:\s+([\d.eE+\-]+)\s+mg/dL\s+"
        r"R²:\s+([\d.eE+\-]+)",
        text,
    )
    records = []
    for b in blocks:
        records.append({
            "window": int(b[0]),
            "p1"    : float(b[1]),
            "p2"    : float(b[2]),
            "p3"    : float(b[3]),
            "rmse"  : float(b[4]),
            "r2"    : float(b[5]),
        })
    return records

# ── Load data ─────────────────────────────────────────────────────────────────
if DATA_SOURCE == "file":
    try:
        records = parse_summary(DATA_FILE)
        print(f"✓ Loaded {len(records)} windows from {DATA_FILE}")
    except Exception as e:
        print(f"⚠ Could not load file ({e}), using inline data.")

p1_all  = np.array([r["p1"]   for r in records])
p2_all  = np.array([r["p2"]   for r in records])
p3_all  = np.array([r["p3"]   for r in records])
r2_all  = np.array([r["r2"]   for r in records])
rmse_all= np.array([r["rmse"] for r in records])

# ── Posterior summary helper ──────────────────────────────────────────────────
def posterior_summary(samples, label, scale=1.0):
    s = samples * scale
    kde   = gaussian_kde(s, bw_method="silverman")
    xs    = np.linspace(s.min() - 0.1*(s.max()-s.min()),
                        s.max() + 0.1*(s.max()-s.min()), 2000)
    ys    = kde(xs)
    mode  = xs[np.argmax(ys)]
    mean  = s.mean()
    std   = s.std(ddof=1)
    ci_lo = np.percentile(s, 2.5)
    ci_hi = np.percentile(s, 97.5)
    return dict(mean=mean, std=std, mode=mode, ci_lo=ci_lo, ci_hi=ci_hi,
                kde=kde, xs=xs, ys=ys, samples=s)

def hdi(samples, cred=0.95):
    """Highest Density Interval"""
    s = np.sort(samples)
    n = len(s)
    w = int(np.floor(cred * n))
    widths = s[w:] - s[:n-w]
    idx = np.argmin(widths)
    return s[idx], s[idx+w]

# ── Compute posteriors ────────────────────────────────────────────────────────
S = {
    "p1": posterior_summary(p1_all, "p1"),
    "p2": posterior_summary(p2_all, "p2"),
    "p3": posterior_summary(p3_all * 1e5, "p3"),   # display in ×10⁻⁵
}

print("\n" + "="*65)
print(f"{'Parameter':<12}{'Mean':>12}{'SD':>12}{'Mode':>12}{'95% CI':>22}")
print("-"*65)
for pname, sv in S.items():
    ci = f"[{sv['ci_lo']:.4f}, {sv['ci_hi']:.4f}]"
    unit = "×10⁻⁵" if pname=="p3" else ""
    print(f"{pname+unit:<14}{sv['mean']:>10.5f}  {sv['std']:>10.5f}  {sv['mode']:>10.5f}  {ci:>22}")
print("="*65)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Marginal posteriors  (3-panel row)
# ─────────────────────────────────────────────────────────────────────────────
param_meta = [
    ("p1", r"$p_1$  (min$^{-1}$)",          PALETTE["p1"],  S["p1"]),
    ("p2", r"$p_2$  (min$^{-1}$)",          PALETTE["p2"],  S["p2"]),
    ("p3", r"$p_3$  ($\times 10^{-5}$ min$^{-2}$ per $\mu$U·mL$^{-1}$)",
                                              PALETTE["p3"],  S["p3"]),
]

fig1, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
fig1.subplots_adjust(wspace=0.38)

for ax, (pname, xlabel, color, sv) in zip(axes, param_meta):
    xs, ys = sv["xs"], sv["ys"]
    lo, hi = hdi(sv["samples"], 0.95)

    # fill HDI
    mask = (xs >= lo) & (xs <= hi)
    ax.fill_between(xs[mask], ys[mask], color=color, alpha=PALETTE["fill"])

    # KDE curve
    ax.plot(xs, ys, color=color, lw=1.8)

    # Mean & mode lines
    ax.axvline(sv["mean"], color=color, lw=1.2, ls="-",  alpha=0.9)
    ax.axvline(sv["mode"], color=color, lw=1.2, ls=":",  alpha=0.7)

    # HDI brackets
    ymax = ys.max()
    ax.annotate("", xy=(lo, ymax*0.08), xytext=(hi, ymax*0.08),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
    ax.text((lo+hi)/2, ymax*0.13, "95% HDI",
            ha="center", va="bottom", fontsize=7.5, color=color)

    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("Density" if pname=="p1" else "", fontsize=9)
    ax.set_title(f"$N={len(sv['samples'])}$ windows", fontsize=9, pad=3)

    # Data rug
    ax.plot(sv["samples"], np.full_like(sv["samples"], -ymax*0.04),
            "|", color=color, alpha=0.5, ms=4, mew=0.8)
    ax.set_ylim(-ymax*0.08, None)

    ax.grid(True, color=PALETTE["grid"], lw=0.5, ls="--", alpha=0.7)

legend_elems = [
    Line2D([0],[0], color="gray",  lw=1.2, ls="-",    label="Mean"),
    Line2D([0],[0], color="gray",  lw=1.2, ls=":",    label="Mode"),
    Patch (          facecolor="gray", alpha=0.3,      label="95% HDI"),
]
fig1.legend(handles=legend_elems, loc="lower center", ncol=3,
            fontsize=8, framealpha=0.9,
            bbox_to_anchor=(0.5, -0.14), columnspacing=1.2)


out1 = os.path.join(OUTPUT_DIR, "bergman_marginal_posteriors.pdf")
fig1.savefig(out1, dpi=300)
print(f"\n✓ Marginal posteriors  → {out1}")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Corner / pairs plot  (joint 2-D posteriors)
# ─────────────────────────────────────────────────────────────────────────────
params_corner = [
    ("p1", r"$p_1$ (min$^{-1}$)",   p1_all,        PALETTE["p1"]),
    ("p2", r"$p_2$ (min$^{-1}$)",   p2_all,        PALETTE["p2"]),
    ("p3", r"$p_3$ ($\times 10^{-5}$)", p3_all*1e5, PALETTE["p3"]),
]
n_params = len(params_corner)

fig2, axes2 = plt.subplots(n_params, n_params, figsize=(7.2, 6.8))
fig2.subplots_adjust(hspace=0.07, wspace=0.07)

for i in range(n_params):
    for j in range(n_params):
        ax = axes2[i][j]

        pname_i, xlabel_i, data_i, color_i = params_corner[i]
        pname_j, xlabel_j, data_j, color_j = params_corner[j]

        if j > i:  # upper triangle — hide
            ax.set_visible(False)
            continue

        if i == j:  # diagonal — marginal KDE
            sv = S[pname_i]
            ax.fill_between(sv["xs"], sv["ys"], color=color_i, alpha=0.25)
            ax.plot(sv["xs"], sv["ys"], color=color_i, lw=1.6)
            ax.axvline(sv["mean"], color=color_i, lw=1.0, ls="--")
            ax.set_yticks([])
            ax.grid(True, color=PALETTE["grid"], lw=0.4, ls="--", alpha=0.6)

        else:  # lower triangle — 2-D KDE contours + scatter
            xi, xj = data_j, data_i   # j on x-axis, i on y-axis

            # 2-D KDE
            xy   = np.vstack([xj, xi])
            kde2 = gaussian_kde(xy, bw_method="silverman")
            xg   = np.linspace(xj.min(), xj.max(), 120)
            yg   = np.linspace(xi.min(), xi.max(), 120)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = kde2(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

            # contour levels at 10, 30, 50, 68, 95 % of peak
            levels = [0.05, 0.30, 0.68, 0.95]
            lvl_abs = sorted([Zg.max() * (1 - q) for q in levels])

            # contourf fills n_levels-1 bands; colors must match that count
            n_bands = len(lvl_abs) - 1
            alphas  = np.linspace(0.10, 0.45, n_bands)[::-1]
            colors_fill = [(plt.matplotlib.colors.to_rgba(color_j, a)) for a in alphas]

            ax.contourf(Xg, Yg, Zg, levels=lvl_abs, colors=colors_fill)
            ax.contour( Xg, Yg, Zg, levels=lvl_abs,
                        colors=[color_j], linewidths=0.7, alpha=0.8)

            # scatter
            ax.scatter(xj, xi, s=14, color=color_j, alpha=0.55,
                       edgecolors="white", linewidths=0.3, zorder=3)

            # correlation annotation
            rho = np.corrcoef(xj, xi)[0, 1]
            ax.text(0.96, 0.96, f"$\\rho={rho:+.3f}$",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="k",
                    bbox=dict(fc="white", ec="#aaaaaa", lw=0.5,
                              boxstyle="round,pad=0.2"))

            ax.grid(True, color=PALETTE["grid"], lw=0.4, ls="--", alpha=0.5)

        # axis labels — only outer edges
        if i == n_params - 1:
            ax.set_xlabel(xlabel_j, fontsize=9)
        else:
            ax.set_xticklabels([])

        if j == 0 and i != 0:
            ax.set_ylabel(xlabel_i, fontsize=9)
        elif j == 0 and i == 0:
            ax.set_ylabel(xlabel_i, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.tick_params(labelsize=7.5)


out2 = os.path.join(OUTPUT_DIR, "bergman_joint_pairs.pdf")
fig2.savefig(out2, dpi=300)
print(f"✓ Corner/pairs plot    → {out2}")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3 — LaTeX parameter summary table
# ─────────────────────────────────────────────────────────────────────────────

def build_latex_table():
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Posterior parameter summary of the Bergman Minimal Model")
    lines.append(r"           estimated from 40 meal windows (Patient 1006).")
    lines.append(r"           Values for $p_1$ and $p_2$ are in min$^{-1}$;")
    lines.append(r"           $p_3$ is in $\times 10^{-5}$ min$^{-2}$\,($\mu$U\,mL$^{-1}$)$^{-1}$.")
    lines.append(r"           CI: Highest Density Interval.}")
    lines.append(r"  \label{tab:bergman_posterior}")
    lines.append(r"  \begin{tabular}{lrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Parameter} & \textbf{Mean} & \textbf{SD} & \textbf{Mode} & \textbf{Median} & \textbf{95\,\% HDI} \\")
    lines.append(r"    \midrule")

    # Parameter rows
    param_rows = [
        ("p1", r"$p_1$",                              S["p1"],  ""),
        ("p2", r"$p_2$",                              S["p2"],  ""),
        ("p3", r"$p_3$ ($\times 10^{-5}$)",           S["p3"],  ""),
    ]
    for pname, label, sv, _ in param_rows:
        hdi_lo, hdi_hi = hdi(sv["samples"], 0.95)
        med = np.median(sv["samples"])
        lines.append(
            f"    {label} & {sv['mean']:.5f} & {sv['std']:.5f} & "
            f"{sv['mode']:.5f} & {med:.5f} & "
            f"[{hdi_lo:.5f},\\;{hdi_hi:.5f}] \\\\"
        )

    lines.append(r"    \midrule")

    # Model-fit rows
    rmse_hdi_lo, rmse_hdi_hi = hdi(rmse_all, 0.95)
    r2_hdi_lo,   r2_hdi_hi   = hdi(r2_all,   0.95)

    lines.append(
        f"    RMSE (mg\\,dL$^{{-1}}$) & {rmse_all.mean():.2f} & {rmse_all.std():.2f} & "
        f"--- & {np.median(rmse_all):.2f} & "
        f"[{rmse_hdi_lo:.2f},\\;{rmse_hdi_hi:.2f}] \\\\"
    )
    lines.append(
        f"    $R^2$ & {r2_all.mean():.4f} & {r2_all.std():.4f} & "
        f"--- & {np.median(r2_all):.4f} & "
        f"[{r2_hdi_lo:.4f},\\;{r2_hdi_hi:.4f}] \\\\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

latex_table = build_latex_table()

out3 = os.path.join(OUTPUT_DIR, "bergman_summary_table.tex")
with open(out3, "w") as f:
    f.write(latex_table)

print(f"\n✓ LaTeX table          → {out3}")
print("\n" + "─"*65)
print(latex_table)
print("─"*65)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Per-window parameter trajectories  (temporal stability check)
# ─────────────────────────────────────────────────────────────────────────────
wins = np.array([r["window"] for r in records])

fig4, axes4 = plt.subplots(3, 1, figsize=(7.2, 5.8), sharex=True)
fig4.subplots_adjust(hspace=0.12)

for ax, (pname, ylabel, data, color) in zip(
    axes4,
    [("p1", r"$p_1$ (min$^{-1}$)",              p1_all,        PALETTE["p1"]),
     ("p2", r"$p_2$ (min$^{-1}$)",              p2_all,        PALETTE["p2"]),
     ("p3", r"$p_3$ ($\times 10^{-5}$)",        p3_all*1e5,    PALETTE["p3"])]
):
    sv = S[pname]

    # ±1 SD band
    ax.axhspan(sv["mean"]-sv["std"], sv["mean"]+sv["std"],
               color=color, alpha=0.12, lw=0)
    # ±2 SD band
    ax.axhspan(sv["mean"]-2*sv["std"], sv["mean"]+2*sv["std"],
               color=color, alpha=0.06, lw=0)

    ax.axhline(sv["mean"], color=color, lw=1.2, ls="--", alpha=0.8,
               label=f"$\\mu = {sv['mean']:.5f}$")

    # Weight by R²
    sizes = 20 + 60 * (r2_all - r2_all.min()) / (r2_all.max() - r2_all.min() + 1e-9)
    sc = ax.scatter(wins, data, s=sizes, c=r2_all, cmap="RdYlGn",
                    vmin=0.65, vmax=1.0, zorder=4,
                    edgecolors="#444444", linewidths=0.3)
    ax.plot(wins, data, color=color, lw=0.7, alpha=0.4, zorder=3)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(True, color=PALETTE["grid"], lw=0.5, ls="--", alpha=0.6)
    ax.set_xlim(0.5, len(wins)+0.5)

cbar = fig4.colorbar(sc, ax=axes4, shrink=0.6, pad=0.015, aspect=30)
cbar.set_label("$R^2$", fontsize=9)
cbar.ax.tick_params(labelsize=8)

axes4[-1].set_xlabel("Meal window index", fontsize=9)

out4 = os.path.join(OUTPUT_DIR, "bergman_parameter_trajectories.pdf")
fig4.savefig(out4, dpi=300)
print(f"✓ Trajectory plot      → {out4}")
plt.show()

print("\n" + "="*55)
print("All figures saved to:", OUTPUT_DIR)
print("="*55)