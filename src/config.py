"""
Configuration for Lorenz Attractor Chaos Prediction.

Centralizes all hyperparameters, system parameters, and paths
to ensure reproducibility and clean experimentation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class LorenzParams:
    """Standard Lorenz system parameters (σ, ρ, β)."""
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


@dataclass(frozen=True)
class SimulationConfig:
    """Controls trajectory generation."""
    t_start: float = 0.0
    t_end: float = 50.0
    dt: float = 0.01
    initial_state: tuple = (1.0, 1.0, 1.0)
    transient_steps: int = 500  # discard initial transient
    noise_std: float = 0.0     # additive Gaussian noise (0 = clean)
    random_seed: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    """Controls Takens embedding and feature engineering."""
    # Takens delay embedding (theoretically: embed_dim >= 2*attractor_dim + 1 = 7)
    embedding_dims: List[int] = field(default_factory=lambda: [3, 5, 7, 10])
    delay_tau: int = 10  # delay in timesteps

    # Rolling statistics windows
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 25, 50])

    # Finite difference features
    diff_orders: List[int] = field(default_factory=lambda: [1, 2])

    # Prediction horizons to evaluate (in timesteps)
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100])


@dataclass(frozen=True)
class ModelConfig:
    """XGBoost and training configuration."""
    # Temporal split ratios
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # XGBoost hyperparameters
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 30,
    })

    # Baseline comparisons
    run_baselines: bool = True  # linear regression, ridge, random forest


@dataclass(frozen=True)
class PathConfig:
    """Project directory layout."""
    root: Path = Path(".")
    figures: Path = Path("figures")
    models: Path = Path("models")
    data: Path = Path("data")

    def ensure_dirs(self):
        for p in [self.figures, self.models, self.data]:
            p.mkdir(parents=True, exist_ok=True)


# ── Global config instances ──────────────────────────────────────────
LORENZ = LorenzParams()
SIM = SimulationConfig()
FEAT = FeatureConfig()
MODEL = ModelConfig()
PATHS = PathConfig()
