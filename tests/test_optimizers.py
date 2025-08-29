
import numpy as np
import pandas as pd
from src.portfolio_app.optimizers import optimize_mv

def test_optimize_mv_returns_weights():
    np.random.seed(0)
    returns = pd.DataFrame(np.random.randn(100, 4), columns=["A", "B", "C", "D"])
    weights = optimize_mv(returns, method="max_sharpe", rfr=0.01, max_w=0.5)
    
    assert isinstance(weights, dict)
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-3)
