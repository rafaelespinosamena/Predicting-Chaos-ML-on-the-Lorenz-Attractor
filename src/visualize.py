"""
Visualization Suite for Chaos Prediction.

Publication-quality figures with a dark aesthetic. Each function
saves a high-res PNG and returns the figure for optional display.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy.typing import NDArray

from src.config import FEAT, PATHS, SIM
from src.train import HorizonResult

logger = logging.getLogger(__name__)

# ── Global style ─────────────────────────────────────────────────────

DARK_BG = "#0a0a0f"
PANEL_BG = "#0e0e18"
ACCENT_COLORS = ["#00d4ff", "#ff6b6b", "#50fa7b", "#bd93f9", "#ffb86c", "#ff79c6"]
GRID_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
SUBTLE_TEXT = "#707090"

def set_style():
    """Apply the dark publication style globally."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "text.color": TEXT_COLOR,
        "xtick.color": SUBTLE_TEXT,
        "ytick.color": SUBTLE_TEXT,
        "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID_COLOR,
        "legend.fontsize": 9,
        "font.family": "monospace",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": DARK_BG,
        "savefig.pad_inches": 0.3,
    })


def _save(fig: plt.Figure, name: str):
    """Save figure to the figures directory."""
    path = PATHS.figures / f"{name}.png"
    fig.savefig(path)
    logger.info(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 1. THE ATTRACTOR — 3D trajectory with time-color gradient
# ══════════════════════════════════════════════════════════════════════

def plot_attractor_3d(
    states: NDArray,
    t: NDArray,
    title: str = "Lorenz Attractor",
    filename: str = "01_attractor_3d",
) -> plt.Figure:
    """
    Stunning 3D plot of the Lorenz attractor, colored by time progression.
    Uses line segments for smooth color gradient along the trajectory.
    """
    set_style()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = states[:, 0], states[:, 1], states[:, 2]

    # Create colored line segments
    points = states.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Time-based coloring
    norm = Normalize(vmin=t[0], vmax=t[-1])
    colors = cm.plasma(norm(t[:-1]))

    # Plot segments
    for i in range(0, len(segments), 3):  # skip some for performance
        ax.plot(
            [segments[i, 0, 0], segments[i, 1, 0]],
            [segments[i, 0, 1], segments[i, 1, 1]],
            [segments[i, 0, 2], segments[i, 1, 2]],
            color=colors[i], alpha=0.7, linewidth=0.4,
        )

    # Glow effect: plot again with thicker, more transparent lines
    subsample = slice(None, None, 10)
    ax.plot(x[subsample], y[subsample], z[subsample],
            color=ACCENT_COLORS[0], alpha=0.03, linewidth=3)

    ax.set_xlabel("X", fontsize=12, labelpad=10)
    ax.set_ylabel("Y", fontsize=12, labelpad=10)
    ax.set_zlabel("Z", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color=ACCENT_COLORS[0])

    # Dark 3D styling
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.view_init(elev=25, azim=135)

    # Colorbar
    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Time (s)", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=SUBTLE_TEXT)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=SUBTLE_TEXT)

    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 2. TIME SERIES — raw x, y, z vs time
# ══════════════════════════════════════════════════════════════════════

def plot_time_series(
    states: NDArray,
    t: NDArray,
    filename: str = "02_time_series",
) -> plt.Figure:
    """Three-panel time series showing the chaotic oscillations."""
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    coords = ["x(t)", "y(t)", "z(t)"]
    colors = [ACCENT_COLORS[0], ACCENT_COLORS[1], ACCENT_COLORS[2]]

    for i, (ax, label, color) in enumerate(zip(axes, coords, colors)):
        ax.plot(t, states[:, i], color=color, alpha=0.85, linewidth=0.5)
        ax.fill_between(t, states[:, i], alpha=0.08, color=color)
        ax.set_ylabel(label, fontsize=12, fontweight="bold")
        ax.set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(
        "Lorenz System — Deterministic Chaos",
        fontsize=15, fontweight="bold", color=ACCENT_COLORS[0], y=0.98,
    )
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 3. PREDICTION vs REALITY — overlay per coordinate
# ══════════════════════════════════════════════════════════════════════

def plot_prediction_vs_actual(
    y_true: NDArray,
    y_pred: NDArray,
    horizon: int,
    t_test: Optional[NDArray] = None,
    filename: str = "03_pred_vs_actual",
) -> plt.Figure:
    """
    Overlay predicted and actual trajectories for test set.
    Shows a zoomed window for detail + full test set for context.
    """
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    coords = ["x", "y", "z"]
    colors_true = [ACCENT_COLORS[0], ACCENT_COLORS[1], ACCENT_COLORS[2]]
    color_pred = ACCENT_COLORS[3]

    n = len(y_true)
    t = t_test if t_test is not None else np.arange(n)
    # Show a detailed window (first 20% of test set)
    window = slice(0, max(n // 5, 200))

    for i, (ax, coord) in enumerate(zip(axes, coords)):
        ax.plot(
            t[window], y_true[window, i],
            color=colors_true[i], alpha=0.9, linewidth=1.2,
            label="Actual", zorder=2,
        )
        ax.plot(
            t[window], y_pred[window, i],
            color=color_pred, alpha=0.7, linewidth=1.0,
            linestyle="--", label="Predicted", zorder=3,
        )
        ax.fill_between(
            t[window],
            y_true[window, i], y_pred[window, i],
            alpha=0.15, color=ACCENT_COLORS[4],
            label="Error", zorder=1,
        )
        ax.set_ylabel(f"{coord}(t+{horizon})", fontsize=11, fontweight="bold")
        if i == 0:
            ax.legend(loc="upper right", ncol=3)

    axes[-1].set_xlabel("Test sample index", fontsize=12)
    fig.suptitle(
        f"XGBoost Prediction vs Ground Truth — Horizon = {horizon} steps",
        fontsize=14, fontweight="bold", color=ACCENT_COLORS[0], y=0.98,
    )
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 4. LYAPUNOV HORIZON — how error grows with prediction horizon
# ══════════════════════════════════════════════════════════════════════

def plot_horizon_analysis(
    results: Dict[str, List[HorizonResult]],
    lyapunov_exp: float,
    dt: float = SIM.dt,
    filename: str = "04_horizon_analysis",
) -> plt.Figure:
    """
    The hero figure: shows MSE and R² vs prediction horizon for all models.
    Overlays the theoretical Lyapunov divergence timescale.
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (model_name, model_results) in enumerate(results.items()):
        horizons = [r.horizon for r in model_results]
        horizon_times = [h * dt for h in horizons]
        mses = [r.mse for r in model_results]
        r2s = [r.r2 for r in model_results]
        color = ACCENT_COLORS[idx]

        display_name = {"xgboost": "XGBoost", "ridge": "Ridge", "random_forest": "Random Forest"}
        label = display_name.get(model_name, model_name)

        ax1.semilogy(horizon_times, mses, "o-", color=color, label=label,
                      linewidth=2, markersize=8, alpha=0.9, zorder=3)
        ax2.plot(horizon_times, r2s, "o-", color=color, label=label,
                 linewidth=2, markersize=8, alpha=0.9, zorder=3)

    # Lyapunov timescale annotation
    lyap_time = 1.0 / lyapunov_exp if lyapunov_exp > 0 else None
    if lyap_time:
        for ax in [ax1, ax2]:
            ax.axvline(
                lyap_time, color=ACCENT_COLORS[4], linestyle=":",
                linewidth=2, alpha=0.8, label=f"Lyapunov time (1/λ ≈ {lyap_time:.2f}s)",
            )

    # Theoretical exponential divergence on MSE plot
    if lyap_time:
        t_theory = np.linspace(0, max(h * dt for h in FEAT.prediction_horizons), 100)
        # Error grows as e^{2λt} (squared because MSE)
        mse_min = min(r.mse for results_list in results.values() for r in results_list)
        theoretical_mse = mse_min * np.exp(2 * lyapunov_exp * t_theory)
        mask = theoretical_mse < 1e4  # clip for display
        ax1.semilogy(
            t_theory[mask], theoretical_mse[mask],
            "--", color=ACCENT_COLORS[5], alpha=0.5, linewidth=1.5,
            label=f"Theoretical: e^{{2λt}}", zorder=1,
        )

    ax1.set_xlabel("Prediction Horizon (seconds)", fontsize=12)
    ax1.set_ylabel("Mean Squared Error (log)", fontsize=12)
    ax1.set_title("Error Growth with Horizon", fontsize=13, fontweight="bold", color=ACCENT_COLORS[0])
    ax1.legend(loc="upper left")

    ax2.set_xlabel("Prediction Horizon (seconds)", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=12)
    ax2.set_title("Prediction Quality vs Horizon", fontsize=13, fontweight="bold", color=ACCENT_COLORS[0])
    ax2.axhline(0, color=SUBTLE_TEXT, linestyle="-", alpha=0.3)
    ax2.legend(loc="lower left")

    fig.suptitle(
        "Chaos Limits Prediction — Lyapunov Timescale Analysis",
        fontsize=15, fontweight="bold", color=ACCENT_COLORS[0], y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 5. RECURSIVE TRAJECTORY — predicted vs actual in 3D
# ══════════════════════════════════════════════════════════════════════

def plot_recursive_trajectory_3d(
    actual: NDArray,
    predicted: NDArray,
    filename: str = "05_recursive_trajectory_3d",
) -> plt.Figure:
    """
    3D plot comparing actual vs recursively-predicted trajectory.
    Color intensity fades as predictions diverge.
    """
    set_style()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    n_pred = len(predicted)
    n_show = min(n_pred, len(actual))

    # Actual trajectory
    ax.plot(
        actual[:n_show, 0], actual[:n_show, 1], actual[:n_show, 2],
        color=ACCENT_COLORS[0], alpha=0.6, linewidth=0.8, label="Actual",
    )

    # Predicted trajectory — colored by step (fading as errors grow)
    for i in range(n_show - 1):
        alpha = max(0.1, 1.0 - i / n_show)
        ax.plot(
            predicted[i:i+2, 0], predicted[i:i+2, 1], predicted[i:i+2, 2],
            color=ACCENT_COLORS[1], alpha=alpha, linewidth=1.0,
        )

    # Mark divergence point (where error exceeds threshold)
    errors = np.linalg.norm(actual[:n_show] - predicted[:n_show], axis=1)
    median_scale = np.median(np.linalg.norm(actual[:n_show], axis=1))
    diverge_idx = np.argmax(errors > 0.1 * median_scale)
    if diverge_idx > 0:
        ax.scatter(
            *predicted[diverge_idx], color=ACCENT_COLORS[4],
            s=100, zorder=5, edgecolors="white", linewidth=1.5,
            label=f"Divergence (~step {diverge_idx})",
        )

    # Start marker
    ax.scatter(
        *predicted[0], color=ACCENT_COLORS[2],
        s=120, zorder=5, marker="*", edgecolors="white",
        label="Start",
    )

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(
        "Recursive Prediction: Chaos Wins Eventually",
        fontsize=14, fontweight="bold", color=ACCENT_COLORS[0], pad=20,
    )
    ax.legend(loc="upper left", fontsize=10)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax.view_init(elev=25, azim=135)

    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 6. ERROR DIVERGENCE — cumulative error over recursive steps
# ══════════════════════════════════════════════════════════════════════

def plot_error_divergence(
    actual: NDArray,
    predicted: NDArray,
    lyapunov_exp: float,
    dt: float = SIM.dt,
    filename: str = "06_error_divergence",
) -> plt.Figure:
    """
    Per-coordinate error over recursive prediction steps,
    with theoretical Lyapunov divergence overlay.
    """
    set_style()
    n = min(len(actual), len(predicted))
    errors = np.abs(actual[:n] - predicted[:n])
    euclidean_error = np.linalg.norm(actual[:n] - predicted[:n], axis=1)
    t = np.arange(n) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])

    # Top: per-coordinate errors
    coords = ["x", "y", "z"]
    for i, (coord, color) in enumerate(zip(coords, ACCENT_COLORS[:3])):
        ax1.semilogy(t, errors[:, i], color=color, alpha=0.8, linewidth=1.0, label=f"|Δ{coord}|")

    ax1.semilogy(t, euclidean_error, color="white", alpha=0.9, linewidth=2.0, label="‖Δr‖ (Euclidean)")

    # Theoretical divergence
    initial_error = euclidean_error[1] if euclidean_error[1] > 0 else 1e-6
    theoretical = initial_error * np.exp(lyapunov_exp * t)
    ax1.semilogy(t, theoretical, "--", color=ACCENT_COLORS[4], alpha=0.6, linewidth=2,
                  label=f"Theory: ε₀·e^{{λt}} (λ={lyapunov_exp:.2f})")

    ax1.set_ylabel("Prediction Error (log scale)", fontsize=12)
    ax1.set_title(
        "Error Growth in Recursive Prediction — Exponential Divergence",
        fontsize=14, fontweight="bold", color=ACCENT_COLORS[0],
    )
    ax1.legend(loc="upper left", fontsize=10)

    # Bottom: normalized error (fraction of attractor scale)
    attractor_scale = np.std(actual[:n], axis=0).mean()
    normalized_error = euclidean_error / attractor_scale

    ax2.fill_between(t, 0, normalized_error, alpha=0.3, color=ACCENT_COLORS[0])
    ax2.plot(t, normalized_error, color=ACCENT_COLORS[0], linewidth=1.5)
    ax2.axhline(1.0, color=ACCENT_COLORS[1], linestyle="--", alpha=0.5, label="Attractor scale")
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Normalized Error", fontsize=12)
    ax2.set_title("Error Relative to Attractor Scale", fontsize=12, color=ACCENT_COLORS[0])
    ax2.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE — top features driving predictions
# ══════════════════════════════════════════════════════════════════════

def plot_feature_importance(
    result: HorizonResult,
    top_n: int = 25,
    filename: str = "07_feature_importance",
) -> plt.Figure:
    """Horizontal bar chart of top XGBoost feature importances."""
    set_style()
    if result.feature_importances is None:
        logger.warning("No feature importances available.")
        return None

    # Sort and take top N
    indices = np.argsort(result.feature_importances)[-top_n:]
    names = [result.feature_names[i] for i in indices]
    importances = result.feature_importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by feature category
    category_colors = {
        "lag": ACCENT_COLORS[0],
        "diff": ACCENT_COLORS[1],
        "rmean": ACCENT_COLORS[2],
        "rstd": ACCENT_COLORS[2],
        "rmin": ACCENT_COLORS[2],
        "rmax": ACCENT_COLORS[2],
        "theta": ACCENT_COLORS[3],
        "phi": ACCENT_COLORS[3],
        "sin": ACCENT_COLORS[3],
        "cos": ACCENT_COLORS[3],
        "r_xy": ACCENT_COLORS[3],
    }

    bar_colors = []
    for name in names:
        color = ACCENT_COLORS[4]  # default
        for key, c in category_colors.items():
            if key in name:
                color = c
                break
        # Current state features
        if name in ("x", "y", "z", "xy", "xz", "yz", "x2", "y2", "z2", "r"):
            color = ACCENT_COLORS[5]
        bar_colors.append(color)

    bars = ax.barh(range(len(names)), importances, color=bar_colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Features — XGBoost (h={result.horizon})",
        fontsize=14, fontweight="bold", color=ACCENT_COLORS[0],
    )

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT_COLORS[0], label="Takens Embedding"),
        Patch(facecolor=ACCENT_COLORS[1], label="Derivatives"),
        Patch(facecolor=ACCENT_COLORS[2], label="Rolling Stats"),
        Patch(facecolor=ACCENT_COLORS[3], label="Geometric"),
        Patch(facecolor=ACCENT_COLORS[5], label="Current State / Interactions"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 8. MODEL COMPARISON HEATMAP
# ══════════════════════════════════════════════════════════════════════

def plot_model_comparison_heatmap(
    results: Dict[str, List[HorizonResult]],
    filename: str = "08_model_comparison",
) -> plt.Figure:
    """Heatmap: R² score by model × horizon × coordinate."""
    set_style()

    model_names = list(results.keys())
    horizons = [r.horizon for r in results[model_names[0]]]
    coords = ["x", "y", "z"]

    # Build matrix: rows = (model, coord), cols = horizons
    row_labels = []
    data = []
    for model_name in model_names:
        display = {"xgboost": "XGB", "ridge": "Ridge", "random_forest": "RF"}.get(model_name, model_name)
        for coord in coords:
            row_labels.append(f"{display} — {coord}")
            row_data = [r.r2_per_coord[coord] for r in results[model_name]]
            data.append(row_data)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.0)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h={h}" for h in horizons], fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Prediction Horizon (steps)", fontsize=12)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(horizons)):
            val = data[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("R² Score", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=SUBTLE_TEXT)
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color=SUBTLE_TEXT)

    ax.set_title(
        "Model × Coordinate × Horizon — R² Score",
        fontsize=14, fontweight="bold", color=ACCENT_COLORS[0],
    )
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 9. SENSITIVITY TO INITIAL CONDITIONS
# ══════════════════════════════════════════════════════════════════════

def plot_sensitivity(
    states: NDArray,
    t: NDArray,
    perturbation: float = 1e-6,
    filename: str = "09_sensitivity",
) -> plt.Figure:
    """
    Classic butterfly effect demonstration:
    two trajectories from nearly identical initial conditions diverge.
    """
    from scipy.integrate import solve_ivp
    from src.config import LORENZ

    set_style()

    # Reintegrate with a tiny perturbation
    ic_perturbed = np.array(SIM.initial_state) + perturbation
    t_span = (SIM.t_start, SIM.t_end)
    t_eval = np.arange(SIM.t_start, SIM.t_end, SIM.dt)

    sol = solve_ivp(
        lambda t, s: [
            LORENZ.sigma * (s[1] - s[0]),
            s[0] * (LORENZ.rho - s[2]) - s[1],
            s[0] * s[1] - LORENZ.beta * s[2],
        ],
        t_span, ic_perturbed, t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12,
    )
    states_perturbed = sol.y[:, SIM.transient_steps:].T
    t_plot = t

    n = min(len(states), len(states_perturbed))
    divergence = np.linalg.norm(states[:n] - states_perturbed[:n], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[2, 1])

    # Top: overlay x(t) for both
    ax = axes[0]
    ax.plot(t_plot[:n], states[:n, 0], color=ACCENT_COLORS[0], alpha=0.9, linewidth=0.7, label="Original")
    ax.plot(t_plot[:n], states_perturbed[:n, 0], color=ACCENT_COLORS[1], alpha=0.9, linewidth=0.7,
            label=f"Perturbed (Δ={perturbation:.0e})")
    ax.set_ylabel("x(t)", fontsize=12)
    ax.set_title(
        f"Butterfly Effect — Sensitivity to Initial Conditions (Δ₀ = {perturbation:.0e})",
        fontsize=14, fontweight="bold", color=ACCENT_COLORS[0],
    )
    ax.legend(loc="upper right", fontsize=10)

    # Bottom: divergence over time
    ax2 = axes[1]
    ax2.semilogy(t_plot[:n], divergence, color=ACCENT_COLORS[4], linewidth=1.5)
    ax2.fill_between(t_plot[:n], divergence, alpha=0.2, color=ACCENT_COLORS[4])
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("‖Δ‖ (log)", fontsize=12)
    ax2.set_title("Trajectory Divergence", fontsize=12, color=ACCENT_COLORS[4])

    fig.tight_layout()
    _save(fig, filename)
    return fig


# ══════════════════════════════════════════════════════════════════════
# 10. PHASE PORTRAITS — 2D projections
# ══════════════════════════════════════════════════════════════════════

def plot_phase_portraits(
    states: NDArray,
    t: NDArray,
    filename: str = "10_phase_portraits",
) -> plt.Figure:
    """Three 2D phase portraits: (x,y), (x,z), (y,z) with density coloring."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    projections = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]

    for ax, (i, j, xlabel, ylabel) in zip(axes, projections):
        # Use hexbin for density visualization
        hb = ax.hexbin(
            states[:, i], states[:, j],
            gridsize=80, cmap="magma", mincnt=1,
            linewidths=0.1, edgecolors="none",
        )
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(f"{xlabel}–{ylabel} Phase Portrait", fontsize=12, color=ACCENT_COLORS[0])
        ax.set_aspect("auto")

    fig.suptitle(
        "Phase Space Density — Lorenz Attractor Projections",
        fontsize=15, fontweight="bold", color=ACCENT_COLORS[0], y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename)
    return fig
