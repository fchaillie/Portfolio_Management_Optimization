import numpy as np
import pandas as pd
from portfolio_app.metrics import port_metrics


def test_port_metrics_returns_dict():
    """
    Test that port_metrics returns a dictionary with expected keys.
    """
    np.random.seed(0)
    rets = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    weights = {"A": 0.3, "B": 0.4, "C": 0.3}

    metrics = port_metrics(rets, weights, rfr=0.01)

    assert isinstance(metrics, dict)
    assert "sharpe" in metrics  # Expected key
    assert "ann_vol" in metrics  # Annual volatility
    assert "ann_return" in metrics  # Annual return
