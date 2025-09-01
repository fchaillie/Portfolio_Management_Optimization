"""
Streamlit entry point for the Portfolio Optimization Dashboard.
Wires together the UI controls, data fetching, optimization, metrics, and plots.
"""

from __future__ import annotations

import os
import sys
import math
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go


# 🔑 Load API keys / config from environment variables
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "yahoo")

# Ensure project ROOT is importable (so "src" package can be found when running via `streamlit run app/main.py`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from our package
from src.portfolio_app.ui import (
    inject_global_css_banner_and_disclaimer,
    apply_background_image,
    set_sidebar_width,
)
from src.portfolio_app.data import get_price_data, daily_returns
from src.portfolio_app.optimizers import (
    optimize_mv,
    optimize_target_vol,
    optimize_min_cvar,
    optimize_hrp,
    HAS_CVXPY,
    HAS_RISKFOLIO,
)
from src.portfolio_app.metrics import (
    sample_frontier,
    port_metrics,
    hist_var_cvar,
    mc_stats_only,
)
from src.portfolio_app.backtest import backtest
from src.portfolio_app.callbacks import current_list, cb_add, cb_remove

# ---------- One-shot global UI ----------
st.set_page_config(
    page_title="Design your investment strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css_banner_and_disclaimer()
apply_background_image()  # looks for assets/background.jpg
set_sidebar_width(520)  # widen sidebar


# ---------- Session bootstrapping ----------
def init_state() -> None:
    """
    Initialize session_state defaults so widgets have predictable keys.
    """
    st.session_state.setdefault(
        "tickers_text",
        "\n".join(sorted(["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"])),
    )
    st.session_state.setdefault("new_ticker", "")
    st.session_state.setdefault("to_remove", "(none)")
    st.session_state.setdefault("results", None)


init_state()

# ---------- Sidebar ----------
with st.sidebar:
    colA, colB = st.columns([1, 3], gap="small")
    with colA:
        st.text_area(
            "Tickers",
            key="tickers_text",
            height=200,
            placeholder="AAPL\nAMZN\nGOOGL\nMETA\nMSFT\nNVDA\n...",
        )
    with colB:
        q1, q2 = st.columns(2, gap="small")
        with q1:
            st.text_input(
                "Quick add", key="new_ticker", placeholder="e.g., AAPL or NESN.SW"
            )
            st.button("Add", on_click=cb_add, use_container_width=True)
        with q2:
            st.selectbox("Remove ticker", ["(none)"] + current_list(), key="to_remove")
            st.button("Remove selected", on_click=cb_remove, use_container_width=True)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input(
            "Optimization Backtest Start date",
            value=date.today() - timedelta(days=365 * 3),
        )
    with d2:
        end_date = st.date_input("Optimization Backtest End date", value=date.today())

    o1, o2 = st.columns(2, gap="small")
    with o1:
        modes = ["Max Sharpe Ratio", "Min Volatility"]
        if HAS_CVXPY:
            modes += ["Target Volatility", "Min Conditional VaR"]
        if HAS_RISKFOLIO:
            modes += ["Hierarchical Risk Parity"]
        opt_mode = st.selectbox("Optimization mode", modes)
        risk_free_rate_percent = st.number_input(
            "Risk-free rate (annual, %)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            format="%.1f",
        )
        risk_free_rate = risk_free_rate_percent / 100.0
        rebal_choice = st.selectbox(
            "Rebalancing frequency", ["Monthly", "Quarterly"], index=0
        )
    with o2:
        max_w_asset_percent = st.slider(
            "Max weight per asset (%)",
            min_value=5,
            max_value=100,
            value=100,
            step=1,
            format="%.0f",
        )
        max_w_asset = max_w_asset_percent / 100.0
        target_vol_percent = st.slider(
            "Target volatility (annualized, %)",
            min_value=5,
            max_value=40,
            value=15,
            step=1,
            format="%.0f",
        )
        target_vol = target_vol_percent / 100.0
        beta_cvar_percent = st.slider(
            "VaR/CVaR confidence level (%)",
            min_value=80.0,
            max_value=99.5,
            value=95.0,
            step=0.5,
            format="%.1f",
        )
        beta_cvar = beta_cvar_percent / 100.0

    m1, m2 = st.columns(2, gap="small")
    with m1:
        mc_choice = st.selectbox(
            "Investment horizon", ["Monthly", "Quarterly", "Annual"], index=0
        )
        horizon = {"Monthly": 21, "Quarterly": 63, "Annual": 252}[mc_choice]
        months_of_horizon = {21: 1, 63: 3, 252: 12}[horizon]
    with m2:
        txn_cost_bps = st.slider(
            "Transaction cost (bps per rebalancing)", 0.0, 50.0, 5.0, 0.5
        )

    frontier_n = 25
    mc_paths = 100

    # Derived
    mc_rebal_days = 21 if rebal_choice == "Monthly" else 63
    bt_freq = "M" if rebal_choice == "Monthly" else "Q"

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ---------- Compute on demand ----------
if run_btn:
    tickers = sorted(set(current_list()))
    if not tickers:
        st.markdown(
        """
        <div style="background-color:white; padding:10px; border-radius:5px; 
                    font-weight:bold; color:black; text-align:center;">
            Your ticker list is empty
        </div>
        """,
        unsafe_allow_html=True
        )
        st.stop()

    with st.spinner("Fetching data & computing..."):
        # Fetch prices and compute daily returns
        prices = get_price_data(
            tickers,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
        )
        if prices.empty:
            st.markdown(
            """
            <div style="background-color:white; padding:10px; border-radius:5px; 
                        font-weight:bold; color:black; text-align:center;">
                No data fetched. Check tickers or dates
            </div>
            """,
            unsafe_allow_html=True
            )
            st.stop()
        rets = daily_returns(prices).replace([np.inf, -np.inf], pd.NA).dropna(how="any")

        # Equally weighted starting portfolio performance
        n = len(prices.columns)
        weights_eq = np.repeat(1/n, n)
        eq_portfolio_returns = rets.dot(weights_eq)
        cum_return = (1 + eq_portfolio_returns).prod() - 1
        eq_perf_value = round(cum_return * 100, 1)
        
        if rets.shape[0] < 5:
            st.markdown(
            """
            <div style="background-color:white; padding:10px; border-radius:5px; 
                        font-weight:bold; color:black; text-align:center;">
                Please choose a period bigger than a week.
            </div>
            """,
            unsafe_allow_html=True
            )
            st.stop()

        # Choose optimizer
        try:
            if opt_mode == "Max Sharpe Ratio":
                weights = optimize_mv(
                    rets, "max_sharpe", rfr=risk_free_rate, max_w=max_w_asset
                )
            elif opt_mode == "Min Volatility":
                weights = optimize_mv(
                    rets, "min_volatility", rfr=risk_free_rate, max_w=max_w_asset
                )
            elif opt_mode == "Target Volatility":
                if not HAS_CVXPY:
                    st.error("cvxpy not installed.")
                    st.stop()
                weights = optimize_target_vol(
                    rets, target_vol=target_vol, max_w=max_w_asset
                )
            elif opt_mode == "Min Conditional VaR":
                if not HAS_CVXPY:
                    st.error("cvxpy not installed.")
                    st.stop()
                weights = optimize_min_cvar(rets, beta=beta_cvar, max_w=max_w_asset)
            elif opt_mode == "Hierarchical Risk Parity":
                if not HAS_RISKFOLIO:
                    st.error("riskfolio-lib not installed.")
                    st.stop()
                weights = optimize_hrp(rets)
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            st.stop()

        # Metrics, frontier, risk, Monte Carlo, and backtest
        metrics = port_metrics(rets, weights, rfr=risk_free_rate)
        frontier = sample_frontier(rets, n_points=frontier_n)
        vr = hist_var_cvar(rets, weights, alpha=0.95)
        mc_stats = mc_stats_only(
            rets,
            weights,
            n_paths=mc_paths,
            horizon_days=horizon,
            rebalance_every=mc_rebal_days,
            txn_cost_bps=txn_cost_bps,
        )
        eq_bt, summ_bt = backtest(
            prices, weights, freq=bt_freq, txn_cost_bps=txn_cost_bps
        )


    # Store results for sticky display
    st.session_state.results = {
        "prices": prices,
        "frontier": frontier,
        "weights": weights,
        "metrics": metrics,
        "vr": vr,
        "mc_stats": mc_stats,
        "eq_bt": eq_bt,
        "summ_bt": summ_bt,
        "horizon": horizon,
        "rebal_choice": rebal_choice,
        "eq_perf_value": eq_perf_value,
    }

# ---------- Display ----------
res = st.session_state.results
if res is None:
    # Friendly instructions card

    st.markdown(
        """

    
        <div style="display:flex; justify-content:center; margin-top: 10px;">
          <div style="background-color: yellow; padding: 20px; font-size: 1.6rem; font-weight: bold; color: black;
                      border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: left; display: table;">
            1) Enter the tickers of the stocks you're open to invest in<br>
            2) Choose your objectives, risk appetite and investment horizon<br>
            3) Click "Run analysis"<br>
            4) Get your ideal stock portfolio to invest in !
          </div>
        </div>

        <div style="display:flex; justify-content:center; margin-top: 4px;">
            <div style="background-color: black; padding: 20px; font-size: 1rem; font-weight: bold; color: white;
                        border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            All investments carry risk and past performance is not indicative of future results. <br>
            Data Source: Yahoo Finance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


else:
    prices, frontier, weights = res["prices"], res["frontier"], res["weights"]
    metrics, vr, mc_stats = res["metrics"], res["vr"], res["mc_stats"]
    eq_bt, summ_bt = res["eq_bt"], res["summ_bt"]
    horizon, rebal_choice, eq_perf_value = res["horizon"], res["rebal_choice"],res["eq_perf_value"]

    # Title of first graph showing equally weighted starting potfolio backtest performance
    st.markdown(
    f"""
    <div style="background: white; padding: 4px 12px; margin: 4px 0 18px 0; text-align: center;
                font-size: 2.0rem; font-weight: bold; color: black; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
        Chosen portfolio equally weighted<br> backtest performance: {eq_perf_value}%
    </div>
    """,
    unsafe_allow_html=True,
    )    

    fig = go.Figure()
    for c in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[c], mode="lines", name=c))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price", margin=dict(t=10, b=40, l=60, r=20)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin-top: 0px;">
            <div style="background-color: white; padding: 0px 3px; font-size: 1rem; font-weight: bold; color: black;
                        border-radius: 0px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            This graph represents how each stock that you have initially chosen have done over the period you chose.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Frontier
    st.markdown(
        """
        <div style="background: white; padding: 4px 12px; margin: 60px 0 18px 0; text-align: center;
                    font-size: 2.0rem; font-weight: bold; color: black; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
          All optimal portfolios before choosing optimization mode
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not frontier.empty:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=frontier["vol"] * 100,
                y=frontier["ret"] * 100,
                mode="markers",
                name="Frontier samples",
            )
        )
        fig2.update_layout(
            xaxis_title="Annualized volatility (%)",
            yaxis_title="Annualized return (%)",
            margin=dict(t=10, b=40, l=60, r=20),
            xaxis=dict(
                tickformat=".0f",
                showgrid=True,
                dtick=5,
                zeroline=False,
                zerolinewidth=1,
                zerolinecolor="black",
            ),
            yaxis=dict(
                tickformat=".0f",
                showgrid=True,
                dtick=5,
                range=[
                    min(0, frontier["ret"].min() * 100 - 5),
                    (frontier["ret"].max() * 100 + 5),
                ],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="black",
            ),
            plot_bgcolor="white",
        )
        st.plotly_chart(
            fig2, use_container_width=True, config={"displayModeBar": False}
        )
    else:
        st.info("Frontier sampling produced no points (possibly due to data issues).")

    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin-top: 0px;">
            <div style="background-color: white; padding: 0px 3px; font-size: 1rem; font-weight: bold; color: black;
                        border-radius: 0px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            This graph represents some of the optimal portfolios that are availbale to you with the stocks you entered<br>without the options you chose. 
            It's a selection of optimal portfolios with different mixes of risk and return.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weights of {opt_mode} portfolio
    st.markdown(
        f"""
        <div style="background: white; padding: 4px 12px; margin: 60px 0 4px 0; text-align: center;
                    font-size: 2rem; font-weight: bold; color: black; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
            Weights of {opt_mode} portfolio
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Build weights_df (sorted by weight DESC) ---
    if isinstance(weights, pd.Series):
        # Series -> DataFrame with explicit columns
        weights_df = weights.rename_axis("Ticker").reset_index(name="Weight")
    elif isinstance(weights, dict):
        # Dict -> DataFrame
        weights_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
    else:
        st.warning("⚠️ No weights found. Please run the analysis.")
        weights_df = pd.DataFrame(columns=["Ticker", "Weight"])

    # Ensure numeric, drop zero weights, then sort DESC **before** formatting to %
    weights_df["Weight"] = pd.to_numeric(weights_df["Weight"], errors="coerce").fillna(
        0.0
    )
    weights_df = (
        weights_df[weights_df["Weight"] > 0]
        .sort_values("Weight", ascending=False, kind="mergesort")  # stable sort
        .reset_index(drop=True)
    )

    # Now format for display
    weights_df["Weight"] = (weights_df["Weight"] * 100).round(1).astype(str) + "%"

    # Render as styled HTML (unchanged)
    st.markdown(
        """
        <style>
        .custom-table table { width: 100%; border-collapse: collapse; }
        .custom-table th, .custom-table td { text-align: center; padding: 4px 8px; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="custom-table" style="
            background-color: white;
            padding: 0;
            border-radius: 0px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            font-size: 1.0rem;
            margin-bottom: 8px;
            max-height: 360px;
            overflow-y: auto;
        ">
            {weights_df.to_html(index=False, border=0)}
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin-top: 0px;">
            <div style="background-color: white; padding: 0px 3px; font-size: 1rem; font-weight: bold; color: black;
                        border-radius: 0px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            This table represents the weights of the optimal portfolio with the stocks AND options you entered.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Backtest
    st.markdown(
        f"""
        <div style="background: white; padding: 4px 12px; margin: 60px 0 18px 0; text-align: center;
                    font-size: 2.0rem; font-weight: bold; color: black; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
          {opt_mode} Portfolio Backtest:<br>Target with {rebal_choice} rebalancing vs Buy & Hold
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq_bt.index,
            y=eq_bt["Target"] * 100,
            mode="lines",
            name="Target (Rebalanced)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=eq_bt.index, y=eq_bt["Buy & Hold"] * 100, mode="lines", name="Buy & Hold"
        )
    )
    y_max = math.ceil(eq_bt.max().max() * 100 / 100.0) * 100
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Equity Value (Base 100)",
        yaxis=dict(
            range=[0, y_max],
            #tickvals=list(range(0, y_max + 100, 100)),
            tickformat=".0f",
            showgrid=True,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="black",
        ),
        margin=dict(t=10, b=40, l=60, r=20),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin-top: 0px;">
            <div style="background-color: white; padding: 0px 3px; font-size: 1rem; font-weight: bold; color: black;
                        border-radius: 0px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            This graph represents the weights of the optimal portfolio for the stocks and the options you chose.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics tables
    st.markdown(
        f"""
        <div style="background: white; padding: 4px 12px; margin: 60px 0 0px 0; text-align: center;
                    font-size: 2.0rem; font-weight: bold; color: black; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
          {opt_mode} Portfolio Metrics
        </div>
        """,
        unsafe_allow_html=True,
    )

    def fp(x):
        return f"{x:.1%}"

    left = [
        (f"Total Return (Target, {rebal_choice} rebal.)", fp(summ_bt["TotalReturn_Target"])),
        ("Total Return (Buy & Hold)", fp(summ_bt["TotalReturn_Buy&Hold"])),
        ("Annualized Return", fp(metrics["ann_return"])),
        ("Annualized Volatility", fp(metrics["ann_vol"])),
        ("Sharpe Ratio", f"{metrics['sharpe']:.2f}"),
        ("Daily VaR (95%)", fp(vr["VaR"])),
        ("Daily CVaR (95%)", fp(vr["CVaR"]))]
    right = [
        (f"Monte Carlo simulations return prob. 5% ({months_of_horizon} M)",fp(mc_stats["p5"])),
        (f"Monte Carlo simulations return prob. 50% ({months_of_horizon} M)",fp(mc_stats["p50"])),
        (f"Monte Carlo simulations return prob. 95% ({months_of_horizon} M)",fp(mc_stats["p95"])),
        (f"Monte Carlo simulations Expected return ({months_of_horizon} M)",fp(mc_stats["mean"])),
        (f"Monte Carlo simulations std ({months_of_horizon} M)", fp(mc_stats["std"]))]

    cL, cR = st.columns(2)
    with cL:
        table_left = pd.DataFrame(left, columns=["Backtest metrics", "Value"])
        st.markdown(
            """<style>.custom-table table { width: 100%; border-collapse: collapse; }
                           .custom-table th, .custom-table td { text-align: center; padding: 4px 8px; }</style>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="custom-table" style="background-color: white; padding: 0; border-radius: 0px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.10); font-size: 1rem; margin-bottom: 2px;margin-top: 1px;">
                    {table_left.to_html(index=False, border=0)}</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style="display:flex; justify-content:center; margin-top: px;">
            <div style="background-color: white; padding: 0px 0px; font-size: 1rem; font-weight: bold; color: black;
                        border-radius: 0px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            This table shows some metrics related to the backtest of your optimal portfolio for the period you chose.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with cR:
        table_right = pd.DataFrame(right, columns=["Forward looking metrics", "Value"])
        st.markdown(
            """<style>.custom-table table { width: 100%; border-collapse: collapse; }
                           .custom-table th, .custom-table td { text-align: center; padding: 4px 8px; }</style>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class="custom-table" style="background-color: white; padding: 0; border-radius: 0px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.10); font-size: 1rem; margin-bottom: 2px;margin-top: 1px;">
                    {table_right.to_html(index=False, border=0)}</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        <div style="display:flex; justify-content:center; margin-top: 0px;">
            <div style="background-color: white; padding: 0px 0px; font-size: 1rem; font-weight: bold; color: black;
                        border-radius: 0px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; display:inline-block;">
            This table shows some metrics related to the possible future performance of your optimal portfolio.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
