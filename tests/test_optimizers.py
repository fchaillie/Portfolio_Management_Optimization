

import numpy as np
import pandas as pd
from portfolio_app.optimizers import optimize_mv

def test_optimize_mv_returns_weights():
    """
    Test that optimize_mv returns weights summing to ~1 and keys matching the assets.
    """
    np.random.seed(0)
    returns = pd.DataFrame(
        np.random.uniform(-0.05, 0.05, size=(100, 4)),
        columns=["A", "B", "C", "D"]
    )
    
    # Use a low RFR to avoid error from PyPortfolioOpt (some returns are < RFR)
    weights = optimize_mv(returns, objective="max_sharpe", rfr=0.001, max_w=0.5)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == {"A", "B", "C", "D"}        # All assets included
    total_weight = sum(weights.values())
    assert abs(total_weight - 1) < 1e-5                        # Normalized weights