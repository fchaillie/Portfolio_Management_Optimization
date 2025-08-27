"""
Metrics and simulations: efficient frontier sampling, annualized metrics,
simple historical VaR/CVaR, and a bootstrapped Monte Carlo over daily returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

@st.cache_data(show_spinner=False)
def est_mu_S(returns: pd.DataFrame):
    """
    Estimate expected returns (mu) and covariance (S) once, with caching.
    - mu: historical mean return (compounded) using daily returns.
    - S : Ledoit–Wolf shrinkage covariance on daily returns.
    """
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    S  = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
    return mu, S


@st.cache_data(show_spinner=False)
def sample_frontier(returns, n_points=50):
    """
    Sample points on the mean-variance frontier by sweeping target returns.
    Returns a DataFrame with columns ['ret', 'vol', 'sharpe'].
    """
    mu, S = est_mu_S(returns)
    target_returns = np.linspace(float(mu.min()), float(mu.max()), n_points)
    pts = []
    for tr in target_returns:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.efficient_return(target_return=float(tr))
            r, v, sh = ef.portfolio_performance(risk_free_rate=0.0, verbose=False)
            pts.append({"ret": r, "vol": v, "sharpe": sh})
        except Exception:
            continue
    return pd.DataFrame(pts)


def port_metrics(returns: pd.DataFrame, weights: dict, rfr: float = 0.0) -> dict:
    """
    Compute annualized return/vol and Sharpe ratio from daily returns and weights.
    """
    w = np.array([weights.get(c, 0.0) for c in returns.columns])
    pr = (returns.to_numpy() * w).sum(axis=1)
    mu, sd = pr.mean(), pr.std(ddof=1)
    ann_r = (1 + mu) ** 252 - 1
    ann_v = sd * np.sqrt(252)
    return {"ann_return": float(ann_r), "ann_vol": float(ann_v), "sharpe": float((ann_r - rfr) / (ann_v + 1e-12))}

def hist_var_cvar(returns: pd.DataFrame, weights: dict, alpha: float = 0.95) -> dict:
    """
    Historical daily VaR and CVaR for the weighted portfolio at confidence `alpha`.
    """
    w = np.array([weights.get(c, 0.0) for c in returns.columns])
    pr = (returns.to_numpy() * w).sum(axis=1)
    q  = np.quantile(pr, 1 - alpha)
    tail = pr[pr <= q]
    return {"VaR": float(q), "CVaR": float(tail.mean() if len(tail) else q)}

def mc_stats_only(
    returns: pd.DataFrame,
    weights: dict,
    n_paths: int = 200,
    horizon_days: int = 21,
    seed: int = 42,
    rebalance_every: int | None = None,
    txn_cost_bps: float = 0.0,
) -> dict:
    """
    Bootstrap Monte Carlo over daily returns for a limited horizon.
    - Randomly draws 'horizon_days' daily return vectors per path.
    - Optionally rebalances back to target weights every 'rebalance_every' days,
      subtracting a simple turnover * bps cost.
    Returns summary stats (mean/std/p5/p50/p95).
    """
    rng = np.random.default_rng(seed)
    R = returns.to_numpy()
    n_obs = R.shape[0]
    w_t = np.array([weights.get(c, 0.0) for c in returns.columns])
    finals = np.empty(n_paths)

    for p in range(n_paths):
        idx = rng.integers(0, n_obs, size=horizon_days)
        path = R[idx, :]
        w = w_t.copy()
        equity = 1.0
        for t in range(horizon_days):
            equity *= 1 + float(w @ path[t])
            # Drift weights with realized returns
            w = w * (1 + path[t])
            s = w.sum()
            w = w / s if s else w
            # Periodic rebalance (optional)
            if rebalance_every and (t + 1) % rebalance_every == 0:
                turn = np.abs(w_t - w).sum()
                equity *= (1 - turn * txn_cost_bps / 10000)
                w = w_t.copy()
        finals[p] = equity - 1

    return {
        "mean": float(finals.mean()),
        "std": float(finals.std(ddof=1)),
        "p5": float(np.quantile(finals, 0.05)),
        "p50": float(np.quantile(finals, 0.50)),
        "p95": float(np.quantile(finals, 0.95)),
    }