"""
Model Training & Evaluation.

Trains XGBoost (primary) and baseline models (Ridge, Random Forest)
across multiple prediction horizons. Implements proper temporal
cross-validation, feature importance analysis, and multi-step
recursive prediction for trajectory forecasting.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import FEAT, MODEL, PATHS
from src.features import COORD_NAMES, build_features, temporal_train_val_test_split

logger = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """Stores evaluation results for a single prediction horizon."""
    horizon: int
    model_name: str
    mse: float
    mae: float
    r2: float
    mse_per_coord: Dict[str, float]
    r2_per_coord: Dict[str, float]
    train_time_sec: float
    feature_importances: Optional[NDArray] = None
    feature_names: Optional[List[str]] = None


def train_single_horizon(
    states: NDArray,
    horizon: int,
    model_name: str = "xgboost",
) -> Tuple[HorizonResult, object, Dict]:
    """
    Train and evaluate a model for a single prediction horizon.

    Parameters
    ----------
    states : ndarray, shape (N, 3)
    horizon : int
        Prediction horizon in timesteps.
    model_name : str
        One of 'xgboost', 'ridge', 'random_forest'.

    Returns
    -------
    result : HorizonResult
    model : fitted model
    splits : dict with train/val/test data
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name} | horizon={horizon} steps")
    logger.info(f"{'='*60}")

    # Build features for this horizon
    X, y = build_features(states, target_horizon=horizon)
    splits = temporal_train_val_test_split(
        X, y, MODEL.train_ratio, MODEL.val_ratio
    )

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # Create model
    model = _create_model(model_name, X_val, y_val)

    # Train
    t0 = time.time()
    if model_name == "xgboost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Evaluate on test set
    y_pred = model.predict(X_test)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    y_test_arr = y_test.values

    # Overall metrics
    mse = mean_squared_error(y_test_arr, y_pred)
    mae = mean_absolute_error(y_test_arr, y_pred)
    r2 = r2_score(y_test_arr, y_pred, multioutput="uniform_average")

    # Per-coordinate metrics
    target_names = [f"{c}_target" for c in COORD_NAMES]
    mse_per_coord = {}
    r2_per_coord = {}
    for i, coord in enumerate(COORD_NAMES):
        mse_per_coord[coord] = mean_squared_error(y_test_arr[:, i], y_pred[:, i])
        r2_per_coord[coord] = r2_score(y_test_arr[:, i], y_pred[:, i])

    # Feature importances (XGBoost and RF)
    feat_imp = None
    feat_names = list(X_train.columns)
    if model_name == "xgboost":
        feat_imp = model.feature_importances_
    elif model_name == "random_forest":
        # Average across the 3 output estimators
        feat_imp = np.mean(
            [est.feature_importances_ for est in model.estimators_], axis=0
        )

    result = HorizonResult(
        horizon=horizon,
        model_name=model_name,
        mse=mse,
        mae=mae,
        r2=r2,
        mse_per_coord=mse_per_coord,
        r2_per_coord=r2_per_coord,
        train_time_sec=train_time,
        feature_importances=feat_imp,
        feature_names=feat_names,
    )

    logger.info(f"  MSE={mse:.6f} | MAE={mae:.6f} | R²={r2:.6f} | time={train_time:.1f}s")
    for coord in COORD_NAMES:
        logger.info(f"    {coord}: MSE={mse_per_coord[coord]:.6f}, R²={r2_per_coord[coord]:.6f}")

    return result, model, splits


def recursive_multistep_predict(
    model: object,
    states: NDArray,
    start_idx: int,
    n_steps: int,
    horizon: int = 1,
) -> NDArray:
    """
    Generate a multi-step trajectory by recursively feeding predictions back.

    This is the real test of the model: small errors compound exponentially
    in chaotic systems, revealing the effective prediction horizon.

    Parameters
    ----------
    model : fitted model
    states : ndarray, full trajectory for feature context
    start_idx : int
        Index in the trajectory to start predicting from.
    n_steps : int
        Number of recursive prediction steps.
    horizon : int
        The single-step horizon the model was trained on.

    Returns
    -------
    trajectory : ndarray, shape (n_steps, 3)
        Predicted trajectory.
    """
    # We need a mutable copy of the trajectory up to start_idx
    buffer = states[:start_idx + 1].copy()
    predictions = []

    for step in range(n_steps):
        # Build features from the current buffer
        X, _ = build_features(buffer, target_horizon=1)
        if len(X) == 0:
            break

        # Predict next state from the last valid feature row
        last_features = X.iloc[[-1]]
        pred = model.predict(last_features)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        next_state = pred[0]
        predictions.append(next_state)

        # Append prediction to buffer for next iteration
        buffer = np.vstack([buffer, next_state.reshape(1, 3)])

    return np.array(predictions)


def run_multi_horizon_experiment(
    states: NDArray,
) -> Dict[str, List[HorizonResult]]:
    """
    Run the full experiment: train all models across all horizons.

    Returns dict: model_name -> list of HorizonResult.
    """
    results: Dict[str, List[HorizonResult]] = {}

    model_names = ["xgboost"]
    if MODEL.run_baselines:
        model_names.extend(["ridge", "random_forest"])

    for model_name in model_names:
        results[model_name] = []
        for horizon in FEAT.prediction_horizons:
            result, model, _ = train_single_horizon(states, horizon, model_name)
            results[model_name].append(result)

            # Save the XGBoost models
            if model_name == "xgboost":
                model_path = PATHS.models / f"xgb_h{horizon}.joblib"
                joblib.dump(model, model_path)

    return results


def _create_model(name: str, X_val=None, y_val=None):
    """Factory for model creation."""
    if name == "xgboost":
        params = MODEL.xgb_params.copy()
        early_stop = params.pop("early_stopping_rounds", 30)
        return XGBRegressor(
            **params,
            early_stopping_rounds=early_stop,
        )
    elif name == "ridge":
        return MultiOutputRegressor(Ridge(alpha=1.0))
    elif name == "random_forest":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        )
    else:
        raise ValueError(f"Unknown model: {name}")
