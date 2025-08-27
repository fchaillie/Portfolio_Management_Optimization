"""
Simple backtester for a fixed target portfolio against a buy-and-hold baseline.
Rebalances the target portfolio on a schedule inferred from `freq`.
"""
from __future__ import annotations

import pandas as pd

def backtest(prices: pd.DataFrame, weights: dict, freq: str = "M", txn_cost_bps: float = 0.0):
    """
    Parameters
    ----------
    prices : DataFrame of asset prices (columns = tickers)
    weights: dict of target weights (sums to 1)
    freq   : "M" (monthly) or "Q" (quarterly) schedule for rebalancing
    txn_cost_bps: simple transaction cost charged on turnover at each rebalance

    Returns
    -------
    eq : DataFrame with equity curves for Target and Buy & Hold
    summ : dict with total returns for both strategies
    """
    # Rebalance points: month-end (M) or quarter-end (Q)
    pts = prices.resample("M" if freq == "ME" or freq == "M" else "QE").last().index

    # Normalize input weights to align with columns
    w_t = pd.Series({k: float(v) for k, v in weights.items()}, index=prices.columns).fillna(0.0)
    eq_t = 1.0  # equity for Target
    eq_b = 1.0  # equity for Buy & Hold
    w = w_t.copy()

    # Buy & hold shares based on initial weights
    bh = (eq_b * w_t) / prices.iloc[0]

    rows_t, rows_b = [], []
    for t in range(1, len(prices)):
        r = prices.iloc[t] / prices.iloc[t - 1] - 1

        # Target portfolio: apply current weights
        eq_t *= 1 + float((w * r).sum())

        # Drift weights based on realized returns
        w = w * (1 + r)
        s = w.sum()
        w = w / s if s else w

        # Buy & Hold valuation
        eq_b = float((bh * prices.iloc[t]).sum())

        # Rebalance at schedule points
        if prices.index[t] in pts:
            turn = (w_t - w).abs().sum()
            eq_t *= (1 - turn * txn_cost_bps / 10000)
            w = w_t.copy()

        rows_t.append((prices.index[t], eq_t))
        rows_b.append((prices.index[t], eq_b))

    df_t = pd.DataFrame(rows_t, columns=["date", "Target"]).set_index("date")
    df_b = pd.DataFrame(rows_b, columns=["date", "Buy & Hold"]).set_index("date")
    eq = df_t.join(df_b, how="outer")

    summ = {
        "TotalReturn_Target": float(eq["Target"].iloc[-1] / eq["Target"].iloc[0] - 1),
        "TotalReturn_Buy&Hold": float(eq["Buy & Hold"].iloc[-1] / eq["Buy & Hold"].iloc[0] - 1),
    }
    return eq, summ