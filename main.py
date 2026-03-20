"""
Predicting Chaos: Machine Learning on the Lorenz Attractor
═══════════════════════════════════════════════════════════

Can gradient-boosted trees learn the structure of a strange attractor?
This project trains XGBoost (and baselines) to predict future states
of the Lorenz system, then measures how prediction error grows
exponentially at the rate dictated by the maximal Lyapunov exponent.

Usage:
    python main.py              # full pipeline
    python main.py --quick      # fast run (fewer horizons, smaller data)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import FEAT, MODEL, PATHS, SIM
from src.simulate import estimate_lyapunov_exponent, generate_trajectory
from src.features import build_features, temporal_train_val_test_split
from src.train import (
    HorizonResult,
    recursive_multistep_predict,
    run_multi_horizon_experiment,
    train_single_horizon,
)
from src.visualize import (
    plot_attractor_3d,
    plot_time_series,
    plot_prediction_vs_actual,
    plot_horizon_analysis,
    plot_recursive_trajectory_3d,
    plot_error_divergence,
    plot_feature_importance,
    plot_model_comparison_heatmap,
    plot_sensitivity,
    plot_phase_portraits,
)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Chaos Prediction Pipeline")
    parser.add_argument("--quick", action="store_true", help="Fast run with reduced data")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    PATHS.ensure_dirs()

    # Quick mode: fewer horizons and shorter trajectory
    if args.quick:
        logger.info("⚡ Quick mode: reduced data and horizons")
        quick_horizons = [1, 10, 50]
        t_end = 30.0
    else:
        quick_horizons = None
        t_end = SIM.t_end

    # ═════════════════════════════════════════════════════════════════
    # STAGE 1: Generate Trajectory
    # ═════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("STAGE 1: Generating Lorenz Attractor Trajectory")
    logger.info("═" * 60)

    t, states = generate_trajectory(t_end=t_end)
    logger.info(f"Trajectory shape: {states.shape}")
    logger.info(f"State ranges: x=[{states[:,0].min():.1f}, {states[:,0].max():.1f}], "
                f"y=[{states[:,1].min():.1f}, {states[:,1].max():.1f}], "
                f"z=[{states[:,2].min():.1f}, {states[:,2].max():.1f}]")

    # Save raw data
    np.savez(PATHS.data / "trajectory.npz", t=t, states=states)

    # ═════════════════════════════════════════════════════════════════
    # STAGE 2: Estimate Lyapunov Exponent
    # ═════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("STAGE 2: Estimating Maximal Lyapunov Exponent")
    logger.info("═" * 60)

    lyapunov_exp = estimate_lyapunov_exponent(n_steps=5_000 if args.quick else 20_000)
    lyapunov_time = 1.0 / lyapunov_exp if lyapunov_exp > 0 else float("inf")
    logger.info(f"  λ_max = {lyapunov_exp:.4f}")
    logger.info(f"  Lyapunov time = {lyapunov_time:.4f} s")
    logger.info(f"  Prediction horizon limit ≈ {lyapunov_time:.2f} seconds")

    # ═════════════════════════════════════════════════════════════════
    # STAGE 3: Visualization — Attractor & System Dynamics
    # ═════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("STAGE 3: Generating System Visualizations")
    logger.info("═" * 60)

    plot_attractor_3d(states, t)
    plot_time_series(states, t)
    plot_phase_portraits(states, t)
    plot_sensitivity(states, t)

    # ═════════════════════════════════════════════════════════════════
    # STAGE 4: Multi-Horizon Training & Evaluation
    # ═════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("STAGE 4: Training Models Across Prediction Horizons")
    logger.info("═" * 60)

    # Override horizons in quick mode
    if quick_horizons:
        import src.config as cfg
        original_horizons = FEAT.prediction_horizons
        # Monkey-patch for quick mode (frozen dataclass workaround)
        object.__setattr__(cfg.FEAT, "prediction_horizons", quick_horizons)

    all_results = run_multi_horizon_experiment(states)

    # ═════════════════════════════════════════════════════════════════
    # STAGE 5: Hero Visualizations — ML Results
    # ═════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("STAGE 5: Generating ML Result Visualizations")
    logger.info("═" * 60)

    # 5a. Horizon analysis (the hero figure)
    plot_horizon_analysis(all_results, lyapunov_exp)

    # 5b. Prediction vs actual for short horizon (best model)
    xgb_h1_result, xgb_h1_model, h1_splits = train_single_horizon(states, horizon=1, model_name="xgboost")
    X_test, y_test = h1_splits["test"]
    y_pred = xgb_h1_model.predict(X_test)
    plot_prediction_vs_actual(y_test.values, y_pred, horizon=1)

    # 5c. Feature importance
    plot_feature_importance(xgb_h1_result)

    # 5d. Model comparison heatmap
    plot_model_comparison_heatmap(all_results)

    # ═════════════════════════════════════════════════════════════════
    # STAGE 6: Recursive Multi-Step Prediction
    # ═════════════════════════════════════════════════════════════════
    logger.info("\n" + "═" * 60)
    logger.info("STAGE 6: Recursive Multi-Step Trajectory Prediction")
    logger.info("═" * 60)

    # Start from middle of test set for context
    n_total = len(states)
    start_idx = int(n_total * 0.75)
    n_recursive_steps = min(800, n_total - start_idx - 1)

    logger.info(f"  Starting recursive prediction from index {start_idx}")
    logger.info(f"  Predicting {n_recursive_steps} steps ahead")

    predicted_traj = recursive_multistep_predict(
        xgb_h1_model, states, start_idx, n_recursive_steps, horizon=1,
    )

    actual_traj = states[start_idx + 1 : start_idx + 1 + len(predicted_traj)]
    n_compare = min(len(actual_traj), len(predicted_traj))

    plot_recursive_trajectory_3d(actual_traj[:n_compare], predicted_traj[:n_compare])
    plot_error_divergence(actual_traj[:n_compare], predicted_traj[:n_compare], lyapunov_exp)

    # ═════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    logger.info("\n" + "═" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("═" * 60)
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Lyapunov exponent: λ = {lyapunov_exp:.4f}")
    logger.info(f"  Prediction horizon limit: ~{lyapunov_time:.2f}s")

    # Print results table
    logger.info("\n  XGBoost Results by Horizon:")
    logger.info(f"  {'Horizon':>8} {'MSE':>10} {'MAE':>10} {'R²':>8}")
    logger.info(f"  {'-'*40}")
    for r in all_results["xgboost"]:
        logger.info(f"  {r.horizon:>8d} {r.mse:>10.4f} {r.mae:>10.4f} {r.r2:>8.4f}")

    logger.info(f"\n  Figures saved to: {PATHS.figures.absolute()}")
    logger.info(f"  Models saved to: {PATHS.models.absolute()}")

    # Print figure manifest
    fig_files = sorted(PATHS.figures.glob("*.png"))
    logger.info(f"\n  Generated {len(fig_files)} figures:")
    for f in fig_files:
        logger.info(f"    • {f.name}")


if __name__ == "__main__":
    main()
