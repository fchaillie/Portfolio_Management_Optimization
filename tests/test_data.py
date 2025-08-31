import pandas as pd
from portfolio_app.data import get_price_data, daily_returns
from unittest.mock import patch

@patch("src.data.get_price_data")
def test_get_price_data_returns_dataframe(mock_get):
    """
    Test that get_price_data returns a valid DataFrame with the requested tickers.
    Uses mocking so test does not rely on external Yahoo API.
    """
    tickers = ["AAPL", "MSFT"]

    # Fake data for testing
    mock_df = pd.DataFrame({
        "AAPL": [150, 151, 152],
        "MSFT": [250, 252, 255]
    })

    # Make get_price_data() return the fake data
    mock_get.return_value = mock_df

    # Call the function (mocked)
    df = get_price_data(tickers, start="2022-01-01", end="2022-01-10")

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert all(ticker in df.columns for ticker in tickers)


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
