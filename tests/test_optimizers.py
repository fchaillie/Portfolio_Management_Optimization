import numpy as np
import pandas as pd
from portfolio_app.optimizers import optimize_mv


def test_optimize_mv_returns_weights():
    """
    Test that optimize_mv returns weights summing to ~1 and keys matching the assets.
    """
    np.random.seed(0)
    returns = pd.DataFrame(
        np.random.uniform(-0.05, 0.05, size=(100, 4)), columns=["A", "B", "C", "D"]
    )

    
    weights = optimize_mv(returns, objective="max_sharpe", rfr=0.001, max_w=0.5)

    assert isinstance(weights, dict)
    # All assets included
    assert set(weights.keys()) == {"A", "B", "C", "D"}  
    total_weight = sum(weights.values())
    # Normalized weights
    assert abs(total_weight - 1) < 1e-5  
