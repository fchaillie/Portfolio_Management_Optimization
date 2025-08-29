"""
Optimizers: classical mean-variance, target-volatility, CVaR, and HRP.
We detect optional solver libraries and expose HAS_* flags for the UI.
"""

from __future__ import annotations

import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions, EfficientCVaR
import streamlit as st

# Optional dependencies — used to enable certain modes in the UI.
import importlib.util

HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None
HAS_RISKFOLIO = importlib.util.find_spec("riskfolio") is not None


@st.cache_data(show_spinner=False)
def est_mu_S(returns: pd.DataFrame):
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    S = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
    return mu, S


def _mv_setup(returns: pd.DataFrame, max_w: float) -> EfficientFrontier:
    """
    Internal helper to build an EfficientFrontier with shrinkage covariance and L2 regularization.
    """
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    S = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, float(max_w)))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    return ef


def optimize_mv(returns, objective="max_sharpe", rfr=0.01, max_w=1.0):
    """
    Mean-variance optimization:
    - objective="max_sharpe"  -> maximize Sharpe ratio given rfr
    - objective="min_volatility" -> minimize total volatility
    Returns a dict of weights.
    """
    mu, S = est_mu_S(returns)
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, float(max_w)))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    (
        ef.max_sharpe(risk_free_rate=rfr)
        if objective == "max_sharpe"
        else ef.min_volatility()
    )
    return ef.clean_weights(1e-3)


def optimize_target_vol(returns, target_vol=0.15, max_w=1.0):
    """
    Efficient frontier at a specified volatility target.
    Requires only PyPortfolioOpt (no cvxpy-specific calls used).
    """
    mu, S = est_mu_S(returns)
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, float(max_w)))
    ef.efficient_risk(target_volatility=float(target_vol))
    return ef.clean_weights(1e-3)


def optimize_min_cvar(returns, beta=0.95, max_w=1.0):
    """
    Minimize CVaR at confidence level `beta`. Requires cvxpy to be available.
    """
    if not HAS_CVXPY:
        raise ImportError("cvxpy not installed")

    # Lazy import: pay the cost only when this mode is used
    import cvxpy as cp  # noqa: F401  (import ensures solver presence; not used directly)

    mu, _ = est_mu_S(returns)  # reuse cached mean
    ec = EfficientCVaR(mu, returns, beta=beta, weight_bounds=(0.0, float(max_w)))
    ec.min_cvar()
    return ec.clean_weights(1e-3)


def optimize_hrp(returns: pd.DataFrame) -> dict:
    """
    Hierarchical Risk Parity weights (via riskfolio-lib).
    We post-normalize in case of tiny negatives from numerical noise.
    """
    if not HAS_RISKFOLIO:
        raise ImportError("riskfolio-lib not installed")

    # Lazy import: only import when HRP is requested
    import riskfolio as rp

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="ledoit", d=0.94)
    w = port.hrp_optimization(model="Classic", rm="MV")[0].to_dict()
    w = {k: float(v) for k, v in w.items()}
    s = sum(max(v, 0) for v in w.values())
    return {k: (max(v, 0) / s if s else 0.0) for k, v in w.items()}
