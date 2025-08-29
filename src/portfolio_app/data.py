"""
Data utilities: download prices and compute daily returns.
Each function is small, testable, and type-annotated where helpful.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(show_spinner=False)
def get_price_data(tickers, start: str, end: str) -> pd.DataFrame:
    """
    Download (adjusted) close prices for one or many tickers between [start, end).
    Returns a DataFrame of prices (columns=tickers, index=datetime).
    The result is cached by Streamlit for faster reruns.
    """
    data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    prices = (
        data["Close"].copy()
        if isinstance(data, pd.DataFrame) and "Close" in data.columns
        else pd.DataFrame(data)
    )

    # For a single ticker, yfinance returns a 1D series — standardize to DataFrame with the given name.
    if isinstance(tickers, (list, tuple)) and len(tickers) == 1:
        prices.columns = [tickers[0]]

    # Clean up: drop empty columns, forward/backward fill small gaps
    return prices.dropna(axis=1, how="all").ffill().bfill()


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute business-day-aligned daily returns, forward-filling missing prices.
    """
    return prices.asfreq("B").ffill().pct_change(fill_method=None).dropna()
