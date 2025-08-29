
import numpy as np
import pandas as pd
from src.portfolio_app.metrics import port_metrics

def test_port_metrics_returns_dict():
    np.random.seed(0)
    rets = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
    weights = {"A": 0.3, "B": 0.4, "C": 0.3}
    metrics = port_metrics(rets, weights, rfr=0.01)

    assert isinstance(metrics, dict)
    assert "Sharpe" in metrics
    assert "Volatility" in metrics