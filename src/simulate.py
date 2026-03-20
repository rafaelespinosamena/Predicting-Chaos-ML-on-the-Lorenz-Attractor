"""
Lorenz System Simulator.

Generates trajectories from the Lorenz attractor using high-accuracy
ODE integration (RK45). Supports noise injection for robustness studies
and computes the maximal Lyapunov exponent via trajectory divergence.
"""

import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.config import LORENZ, SIM

logger = logging.getLogger(__name__)


def lorenz_system(t: float, state: NDArray, sigma: float, rho: float, beta: float) -> list:
    """Lorenz ODE system: dx/dt = f(x, y, z)."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]


def generate_trajectory(
    t_end: float = SIM.t_end,
    dt: float = SIM.dt,
    initial_state: tuple = SIM.initial_state,
    transient_steps: int = SIM.transient_steps,
    noise_std: float = SIM.noise_std,
    seed: int = SIM.random_seed,
) -> Tuple[NDArray, NDArray]:
    """
    Integrate the Lorenz system and return (time, states).

    Parameters
    ----------
    t_end : float
        Integration endpoint (seconds).
    dt : float
        Timestep for output (dense output interpolation).
    initial_state : tuple
        (x0, y0, z0) starting point.
    transient_steps : int
        Number of initial steps to discard (attractor settling).
    noise_std : float
        Standard deviation of additive Gaussian noise (0 = deterministic).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    t : ndarray, shape (N,)
        Time array after transient removal.
    states : ndarray, shape (N, 3)
        Columns are [x, y, z] on the attractor.
    """
    rng = np.random.default_rng(seed)

    t_span = (SIM.t_start, t_end)
    t_eval = np.arange(SIM.t_start, t_end, dt)

    logger.info(
        f"Integrating Lorenz system: σ={LORENZ.sigma}, ρ={LORENZ.rho}, β={LORENZ.beta:.4f}"
    )
    logger.info(f"  t=[{SIM.t_start}, {t_end}], dt={dt}, {len(t_eval)} steps")

    sol = solve_ivp(
        lorenz_system,
        t_span,
        initial_state,
        args=(LORENZ.sigma, LORENZ.rho, LORENZ.beta),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    t = sol.t[transient_steps:]
    states = sol.y[:, transient_steps:].T  # shape (N, 3)

    if noise_std > 0:
        logger.info(f"  Adding Gaussian noise: σ_noise={noise_std}")
        states = states + rng.normal(0, noise_std, states.shape)

    logger.info(f"  Trajectory: {states.shape[0]} points after transient removal")
    return t, states


def estimate_lyapunov_exponent(
    dt: float = SIM.dt,
    n_steps: int = 50_000,
    perturbation: float = 1e-8,
) -> float:
    """
    Estimate the maximal Lyapunov exponent via trajectory divergence.

    Uses the standard algorithm: evolve two nearby trajectories,
    periodically renormalize the perturbation vector, and average
    the logarithmic divergence rate.

    Returns
    -------
    lambda_max : float
        Estimated maximal Lyapunov exponent (positive => chaos).
    """
    rng = np.random.default_rng(SIM.random_seed)
    x0 = np.array(SIM.initial_state, dtype=np.float64)

    # Let the trajectory settle onto the attractor
    sol = solve_ivp(
        lorenz_system, (0, 20), x0,
        args=(LORENZ.sigma, LORENZ.rho, LORENZ.beta),
        method="RK45", rtol=1e-10, atol=1e-12,
    )
    x0 = sol.y[:, -1]

    # Perturbed initial condition
    delta = rng.standard_normal(3)
    delta = delta / np.linalg.norm(delta) * perturbation
    x1 = x0 + delta

    lyap_sum = 0.0
    renorm_interval = 10  # renormalize every N steps

    for step in range(n_steps):
        t_span = (0, dt * renorm_interval)
        t_eval_local = [dt * renorm_interval]

        sol0 = solve_ivp(
            lorenz_system, t_span, x0,
            args=(LORENZ.sigma, LORENZ.rho, LORENZ.beta),
            t_eval=t_eval_local, method="RK45", rtol=1e-10, atol=1e-12,
        )
        sol1 = solve_ivp(
            lorenz_system, t_span, x1,
            args=(LORENZ.sigma, LORENZ.rho, LORENZ.beta),
            t_eval=t_eval_local, method="RK45", rtol=1e-10, atol=1e-12,
        )

        x0 = sol0.y[:, -1]
        x1 = sol1.y[:, -1]

        delta = x1 - x0
        dist = np.linalg.norm(delta)

        if dist > 0:
            lyap_sum += np.log(dist / perturbation)

        # Renormalize: keep direction, reset magnitude
        x1 = x0 + (delta / dist) * perturbation

    total_time = n_steps * dt * renorm_interval
    lambda_max = lyap_sum / total_time

    logger.info(f"Estimated maximal Lyapunov exponent: λ_max = {lambda_max:.4f}")
    return lambda_max
