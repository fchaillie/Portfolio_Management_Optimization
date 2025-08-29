import pandas as pd
from portfolio_app.data import get_price_data, daily_returns

def test_get_price_data_returns_dataframe():
    tickers = ["AAPL", "MSFT"]
    df = get_price_data(tickers, start="2022-01-01", end="2022-01-10")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(ticker in df.columns for ticker in tickers)

def test_daily_returns_shape():
    dates = pd.date_range(start="2023-01-01", periods=4, freq="B")  # business days
    prices = pd.DataFrame({
        "AAPL": [100, 102, 101, 103],
        "MSFT": [200, 198, 202, 204]
    }, index=dates)
    rets = daily_returns(prices)
    assert isinstance(rets, pd.DataFrame)
    assert rets.shape[0] == prices.shape[0] - 1
