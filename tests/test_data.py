import pandas as pd
from portfolio_app.data import get_price_data, daily_returns


def test_get_price_data_returns_dataframe():
    """
    Test that get_price_data:
    - Returns a non-empty DataFrame
    - Contains the expected ticker columns
    """
    tickers = ["AAPL", "MSFT"]
    df = get_price_data(tickers, start="2022-01-01", end="2022-01-10")

    assert isinstance(df, pd.DataFrame)  # Output should be a DataFrame
    assert not df.empty  # Should not be empty
    assert all(
        ticker in df.columns for ticker in tickers
    )  # All tickers should be in columns


def test_daily_returns_shape():
    """
    Test that daily_returns:
    - Returns a DataFrame with N-1 rows
    - Preserves correct column names
    """
    # Create a fixed 4-day price series on business days
    dates = pd.date_range(start="2023-01-01", periods=4, freq="B")
    prices = pd.DataFrame(
        {"AAPL": [100, 102, 101, 103], "MSFT": [200, 198, 202, 204]}, index=dates
    )

    rets = daily_returns(prices)

    assert isinstance(rets, pd.DataFrame)  # Output type
    assert rets.shape[0] == prices.shape[0] - 1  # Should have N-1 rows
    assert list(rets.columns) == ["AAPL", "MSFT"]  # Columns preserved
