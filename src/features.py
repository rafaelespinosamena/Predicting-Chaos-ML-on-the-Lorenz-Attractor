"""
Feature Engineering for Chaotic Time Series.

Implements Takens delay embedding (motivated by the embedding theorem),
rolling statistics, finite differences, and cross-variable interactions.
All features respect temporal causality — no future leakage.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import FEAT

logger = logging.getLogger(__name__)

COORD_NAMES = ["x", "y", "z"]


def build_features(
    states: NDArray,
    target_horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix and target matrix from raw trajectory.

    Parameters
    ----------
    states : ndarray, shape (N, 3)
        Raw [x, y, z] trajectory on the attractor.
    target_horizon : int
        How many steps ahead to predict.

    Returns
    -------
    X : DataFrame, shape (M, F)
        Feature matrix (M < N due to lag/horizon trimming).
    y : DataFrame, shape (M, 3)
        Target values [x(t+h), y(t+h), z(t+h)].
    """
    df = pd.DataFrame(states, columns=COORD_NAMES)
    feature_frames: List[pd.DataFrame] = []

    # ── 1. Current state (identity features) ─────────────────────────
    feature_frames.append(df[COORD_NAMES].copy())
    logger.info("  [features] Current state: 3 features")

    # ── 2. Takens delay embedding ────────────────────────────────────
    takens_features = _takens_embedding(df, FEAT.delay_tau, FEAT.embedding_dims)
    feature_frames.append(takens_features)
    logger.info(f"  [features] Takens embedding: {takens_features.shape[1]} features")

    # ── 3. Finite differences (velocity, acceleration) ───────────────
    diff_features = _finite_differences(df, FEAT.diff_orders)
    feature_frames.append(diff_features)
    logger.info(f"  [features] Finite differences: {diff_features.shape[1]} features")

    # ── 4. Rolling statistics ────────────────────────────────────────
    rolling_features = _rolling_statistics(df, FEAT.rolling_windows)
    feature_frames.append(rolling_features)
    logger.info(f"  [features] Rolling statistics: {rolling_features.shape[1]} features")

    # ── 5. Cross-variable interactions ───────────────────────────────
    interaction_features = _interactions(df)
    feature_frames.append(interaction_features)
    logger.info(f"  [features] Interactions: {interaction_features.shape[1]} features")

    # ── 6. Geometric features ────────────────────────────────────────
    geo_features = _geometric_features(df)
    feature_frames.append(geo_features)
    logger.info(f"  [features] Geometric: {geo_features.shape[1]} features")

    # Combine and align
    X = pd.concat(feature_frames, axis=1)

    # Target: state at t + horizon
    y = df[COORD_NAMES].shift(-target_horizon)
    y.columns = [f"{c}_target" for c in COORD_NAMES]

    # Drop rows with NaN (from lags and horizon shift)
    valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    logger.info(f"  [features] Final: {X.shape[1]} features, {X.shape[0]} samples")
    return X, y


def _takens_embedding(
    df: pd.DataFrame, tau: int, dims: List[int]
) -> pd.DataFrame:
    """
    Takens delay embedding: x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ).

    By the embedding theorem, for embedding dimension d >= 2*dim(attractor)+1,
    the delay embedding is diffeomorphic to the original attractor.
    """
    max_dim = max(dims)
    features = {}

    for coord in COORD_NAMES:
        for lag_idx in range(1, max_dim):
            shift = lag_idx * tau
            features[f"{coord}_lag_{lag_idx}tau"] = df[coord].shift(shift)

    return pd.DataFrame(features)


def _finite_differences(
    df: pd.DataFrame, orders: List[int]
) -> pd.DataFrame:
    """Numerical derivatives via finite differences (backward-looking)."""
    features = {}

    for coord in COORD_NAMES:
        for order in orders:
            features[f"{coord}_diff{order}"] = df[coord].diff(order)

    return pd.DataFrame(features)


def _rolling_statistics(
    df: pd.DataFrame, windows: List[int]
) -> pd.DataFrame:
    """Rolling mean, std, min, max — all backward-looking (no leakage)."""
    features = {}

    for coord in COORD_NAMES:
        for w in windows:
            roll = df[coord].rolling(window=w, min_periods=w)
            features[f"{coord}_rmean_{w}"] = roll.mean()
            features[f"{coord}_rstd_{w}"] = roll.std()
            features[f"{coord}_rmin_{w}"] = roll.min()
            features[f"{coord}_rmax_{w}"] = roll.max()

    return pd.DataFrame(features)


def _interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-variable products and ratios (motivated by the ODE structure)."""
    return pd.DataFrame({
        # Products that appear in the Lorenz equations
        "xy": df["x"] * df["y"],
        "xz": df["x"] * df["z"],
        "yz": df["y"] * df["z"],
        # Quadratic terms
        "x2": df["x"] ** 2,
        "y2": df["y"] ** 2,
        "z2": df["z"] ** 2,
        # Radius from origin
        "r": np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2),
    })


def _geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features capturing geometric position on the attractor."""
    # Angle in xy-plane (which lobe are we on?)
    theta_xy = np.arctan2(df["y"], df["x"])
    # Distance from z-axis
    r_xy = np.sqrt(df["x"]**2 + df["y"]**2)
    # Elevation angle
    phi = np.arctan2(df["z"], r_xy)

    return pd.DataFrame({
        "theta_xy": theta_xy,
        "r_xy": r_xy,
        "phi": phi,
        "sin_theta": np.sin(theta_xy),
        "cos_theta": np.cos(theta_xy),
    })


def temporal_train_val_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Strictly temporal split — no shuffling, no future leakage.

    Returns dict with keys 'train', 'val', 'test', each containing (X, y).
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X.iloc[:train_end], y.iloc[:train_end]),
        "val": (X.iloc[train_end:val_end], y.iloc[train_end:val_end]),
        "test": (X.iloc[val_end:], y.iloc[val_end:]),
    }

    for name, (Xi, yi) in splits.items():
        logger.info(f"  [{name}] {len(Xi)} samples ({len(Xi)/n*100:.1f}%)")

    return splits
