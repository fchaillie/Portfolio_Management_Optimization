# app.py — long-only optimizer (MV / Target-Vol / Min-CVaR / HRP), daily returns
# Mode sidebar (large), résultats "sticky" (se mettent à jour uniquement sur Run analysis)
# Titres en rectangles blancs à bords droits, tables opaques, background avec overlay sombre 0.10.

import base64
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions, EfficientCVaR

# Optional libs
try:
    import cvxpy as cp  # ensures solver presence for CVaR / target-vol
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False

try:
    import riskfolio as rp
    HAS_RISKFOLIO = True
except Exception:
    HAS_RISKFOLIO = False

# ---------- Page ----------
# st.set_page_config(page_title="Design your investment strategy", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {
    transform: translateY(-72px); /* Pull everything up */
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Shrink left and right margins of main content */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Streamlit tables: make all column headers black + bold */
div[data-testid="stTable"] th {
    color: black !important;
    font-weight: 700 !important;
}
div[data-testid="stDataFrame"] th {
    color: black !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# === Minimal, safe CSS: remove top header/toolbar and top padding only ===
st.markdown("""
<style>
  /* Remove Streamlit's top header/toolbar strip */
  header[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }

  /* Pull main content to the very top */
  .block-container { padding-top: 0 !important; margin-top: 0 !important; }
  .block-container > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Reduce space below subheaders (h2/h3) */
h2, h3 {
    margin-bottom: 4px !important;
}

/* Remove extra margin above Plotly charts */
.element-container:has(div[data-testid="stPlotlyChart"]) {
    margin-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Hide the top-right overflow menu of st.dataframe */
[data-testid="stDataFrameContainer"] .row-widget.stButton {
    display: none !important;
}
[data-testid="stDataFrameContainer"] .stActionButton {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Background & CSS ----------
def _inject_bg_css(b64: str, overlay_rgba="rgba(0,0,0,0.10)", blur_px=0):
    st.markdown(f"""
    <style>
    html, body, .stApp {{ height: 100%; min-height: 100vh; }}
    .stApp {{
      background: url("data:image/jpg;base64,{b64}") center / cover fixed no-repeat;
      position: relative;
    }}
    .stApp::before {{
      content: ""; position: fixed; inset: 0;
      background: {overlay_rgba}; pointer-events: none; z-index:0;
      {'backdrop-filter: blur(' + str(blur_px) + 'px);' if blur_px else ''}
    }}
    .stApp > div {{ position: relative; z-index:1; }}

    /* Titres : rectangles blancs bords droits (compact) */
    h1, h2, h3 {{
      color:#000 !important; background:#fff !important; padding:6px 2px;
      border-radius:0px; display:inline-block; box-shadow:0 2px 6px rgba(0,0,0,0.06);
      margin:2px 0 2px 0;
    }}

    /* Tables opaques et compactes */
    div[data-testid="stTable"] {{
      background:#fff !important; border-radius:0px !important;
      box-shadow:0 2px 8px rgba(0,0,0,0.10) !important;
      padding:4px 6px !important; font-size:1.3rem !important; margin-bottom:8px !important;
    }}
    div[data-testid="stTable"] table {{ background:#fff !important; }}
    div[data-testid="stTable"] th {{ font-weight:700 !important; }}

    /* Réduire les espacements verticaux entre éléments */
    [data-testid="element-container"] {{ margin-bottom: 8px !important; }}
    </style>
    """, unsafe_allow_html=True)

def set_bg_local(path="assets/background.jpg"):
    try:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        _inject_bg_css(b64, overlay_rgba="rgba(0,0,0,0.10)", blur_px=0)
    except Exception:
        pass

set_bg_local()

# ---- Sidebar plus large (≈ 2x) ----
def widen_sidebar(width_px: int = 520):
    st.markdown(f"""
    <style>
      section[data-testid="stSidebar"] {{ width: {width_px}px !important; }}
      div[data-testid="stSidebar"] > div {{ padding-right: 8px; }}
      .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    </style>
    """, unsafe_allow_html=True)

widen_sidebar(520)  # ajustez (480–560 px) selon votre écran

# ---------- Session ----------
def ss_init():
    st.session_state.setdefault("tickers_text", "\n".join(sorted(["AAPL","AMZN","GOOGL","META","MSFT","NVDA"])))
    st.session_state.setdefault("new_ticker", "")
    st.session_state.setdefault("to_remove", "(none)")
    st.session_state.setdefault("results", None)
ss_init()

# ---------- Data & Math ----------
@st.cache_data(show_spinner=False)
def get_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"].copy() if isinstance(data, pd.DataFrame) and "Close" in data.columns else pd.DataFrame(data)
    if isinstance(tickers, (list, tuple)) and len(tickers) == 1:
        prices.columns = [tickers[0]]
    return prices.dropna(axis=1, how="all").ffill().bfill()

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.asfreq("B").ffill().pct_change().dropna()

def optimize_mv(returns, objective="max_sharpe", rfr=0.01, max_w=1.0):
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    S  = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, float(max_w)))
    ef.add_objective(objective_functions.L2_reg, gamma=0.001)
    ef.max_sharpe(risk_free_rate=rfr) if objective=="max_sharpe" else ef.min_volatility()
    return ef.clean_weights(1e-3)

def optimize_target_vol(returns, target_vol=0.15, max_w=1.0):
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    S  = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, float(max_w)))
    ef.efficient_risk(target_volatility=float(target_vol))
    return ef.clean_weights(1e-3)

def optimize_min_cvar(returns, beta=0.95, max_w=1.0):
    if not HAS_CVXPY: raise ImportError("cvxpy not installed")
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    ec = EfficientCVaR(mu, returns, beta=beta, weight_bounds=(0.0, float(max_w)))
    ec.min_cvar(); return ec.clean_weights(1e-3)

def optimize_hrp(returns):
    if not HAS_RISKFOLIO: raise ImportError("riskfolio-lib not installed")
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='ledoit', d=0.94)
    w = port.hrp_optimization(model='Classic', rm='MV')[0].to_dict()
    w = {k: float(v) for k, v in w.items()}
    s = sum(max(v,0) for v in w.values());  w = {k: max(v,0)/s for k,v in w.items()} if s else w
    return w

@st.cache_data(show_spinner=False)
def sample_frontier(returns, n_points=50):
    mu = mean_historical_return(returns, compounding=True, returns_data=True)
    S  = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
    target_returns = np.linspace(float(mu.min()), float(mu.max()), n_points)
    pts = []
    for tr in target_returns:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            ef.efficient_return(target_return=float(tr))
            r,v,sh = ef.portfolio_performance(risk_free_rate=0.0, verbose=False)
            pts.append({"ret":r, "vol":v, "sharpe":sh})
        except Exception:
            continue
    return pd.DataFrame(pts)

def port_metrics(returns, weights, rfr=0.0):
    w = np.array([weights.get(c,0.0) for c in returns.columns])
    pr = (returns.to_numpy() * w).sum(axis=1)
    mu, sd = pr.mean(), pr.std(ddof=1)
    ann_r = (1+mu)**252 - 1; ann_v = sd*np.sqrt(252)
    return {"ann_return":float(ann_r), "ann_vol":float(ann_v), "sharpe":float((ann_r-rfr)/(ann_v+1e-12))}

def hist_var_cvar(returns, weights, alpha=0.95):
    w = np.array([weights.get(c,0.0) for c in returns.columns])
    pr = (returns.to_numpy() * w).sum(axis=1)
    q  = np.quantile(pr, 1-alpha); tail = pr[pr<=q]
    return {"VaR":float(q), "CVaR":float(tail.mean() if len(tail) else q)}

def mc_stats_only(returns, weights, n_paths=200, horizon_days=21, seed=42, rebalance_every=None, txn_cost_bps=0.0):
    rng = np.random.default_rng(seed)
    R = returns.to_numpy(); n_obs = R.shape[0]
    w_t = np.array([weights.get(c,0.0) for c in returns.columns])
    finals = np.empty(n_paths)
    for p in range(n_paths):
        idx = rng.integers(0, n_obs, size=horizon_days); path = R[idx,:]
        w = w_t.copy(); eq=1.0
        for t in range(horizon_days):
            eq *= 1 + float(w @ path[t])
            w  = w*(1+path[t]); s=w.sum();  w /= s if s else 1
            if rebalance_every and (t+1)%rebalance_every==0:
                turn = np.abs(w_t-w).sum(); eq *= (1 - turn*txn_cost_bps/10000); w=w_t.copy()
        finals[p]=eq-1
    return {
        "mean":float(finals.mean()), "std":float(finals.std(ddof=1)),
        "p5":float(np.quantile(finals,0.05)), "p50":float(np.quantile(finals,0.50)),
        "p95":float(np.quantile(finals,0.95))
    }

def backtest(prices, weights, freq="M", txn_cost_bps=0.0):
    pts = prices.resample("M" if freq=="ME" else "QE").last().index
    w_t = pd.Series({k:float(v) for k,v in weights.items()}, index=prices.columns).fillna(0.0)
    eq_t=1.0; eq_b=1.0; w=w_t.copy(); bh=(eq_b*w_t)/prices.iloc[0]
    rows_t=[]; rows_b=[]
    for t in range(1,len(prices)):
        r = prices.iloc[t]/prices.iloc[t-1]-1
        eq_t *= 1 + float((w*r).sum())
        w = w*(1+r); s=w.sum(); w/=s if s else 1
        eq_b = float((bh*prices.iloc[t]).sum())
        if prices.index[t] in pts:
            turn=(w_t-w).abs().sum(); eq_t *= (1 - turn*txn_cost_bps/10000); w=w_t.copy()
        rows_t.append((prices.index[t],eq_t)); rows_b.append((prices.index[t],eq_b))
    df_t = pd.DataFrame(rows_t, columns=["date","Target"]).set_index("date")
    df_b = pd.DataFrame(rows_b, columns=["date","Buy & Hold"]).set_index("date")
    eq = df_t.join(df_b, how="outer")
    summ = {"TotalReturn_Target": float(eq["Target"].iloc[-1]/eq["Target"].iloc[0]-1),
            "TotalReturn_Buy&Hold": float(eq["Buy & Hold"].iloc[-1]/eq["Buy & Hold"].iloc[0]-1)}
    return eq, summ

# ---------- Callbacks ----------
def _current_list():
    return [x.strip().upper() for x in st.session_state["tickers_text"].splitlines() if x.strip()]

def cb_add():
    t = st.session_state.get("new_ticker","").strip().upper()
    if not t: return
    try:
        test = yf.download(t, period="1mo", auto_adjust=True, progress=False)
        if test.empty: return
        cur = _current_list(); cur.append(t)
        st.session_state["tickers_text"] = "\n".join(sorted(set(cur)))
        st.session_state["new_ticker"] = ""
    except Exception:
        pass

def cb_remove():
    tr = st.session_state.get("to_remove","(none)")
    if tr=="(none)": return
    cur = _current_list()
    st.session_state["tickers_text"] = "\n".join(sorted(set([x for x in cur if x!=tr])))
    st.session_state["to_remove"] = "(none)"

# ---------- UI (Sidebar) ----------

st.markdown("""
<div style="text-align: center;">
    <div style="
        background-color: #D0F0C0;
        padding: 18px 40px;
        font-size: 2.5rem;
        font-weight: bold;
        border-radius: 0px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        display: inline-block;
        margin-top: 0;
    ">
    Design your investment strategy !
    </div>
</div>
""", unsafe_allow_html=True)

# CSS: remove sidebar header/arrow + pull content further up
st.markdown("""
<style>
  /* Hide all possible toggle buttons + the header row */
  [data-testid="collapsedControl"], 
  [data-testid="baseButton-toggleSidebar"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarToggle"],
  [data-testid="stIconButton"],
  button[title="Collapse sidebar"],
  button[title="Expand sidebar"],
  button[title="Toggle sidebar"] { display:none !important; }
  section[data-testid="stSidebar"] header { display:none !important; height:0 !important; }

  /* Remove residual paddings/margins on containers */
  section[data-testid="stSidebar"] { padding-top:0 !important; }
  section[data-testid="stSidebar"] > div { padding-top:0 !important; margin-top:0 !important; }
  [data-testid="stSidebarContent"] { padding-top:0 !important; margin-top:0 !important; }

  /* Pull the whole content block upward (tune this if you want more/less) */
  [data-testid="stSidebarContent"] {
    transform: translateY(-54px);  /* try -12 to -24 if you want less/more */
  }

  /* Ensure first widget isn’t adding its own top margin */
  [data-testid="stSidebarContent"] > div:first-child { margin-top:0 !important; padding-top:0 !important; }

  /* Tighten vertical gaps between widgets */
  [data-testid="stSidebarContent"] [data-testid="element-container"] { margin-bottom:6px !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # --- Universe: tickers list (narrow) + actions (right) ---
    colA, colB = st.columns([1, 3], gap="small")
    with colA:
        st.text_area(
            "Tickers",
            key="tickers_text",
            height=200,
            placeholder="AAPL\nAMZN\nGOOGL\nMETA\nMSFT\nNVDA\n...",
        )
    with colB:
        # Row: Quick add (left) and Remove (right) — half/half
        q1, q2 = st.columns(2, gap="small")
        with q1:
            st.text_input("Quick add", key="new_ticker", placeholder="e.g., AAPL or NESN.SW")
            st.button("Add", on_click=cb_add, use_container_width=True)
        with q2:
            st.selectbox("Remove ticker", ["(none)"] + _current_list(), key="to_remove")
            st.button("Remove selected", on_click=cb_remove, use_container_width=True)

    # --- Dates (tight two columns) ---
    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Optimization Backtest Start date", value=date.today() - timedelta(days=365*3))
    with d2:
        end_date = st.date_input("Optimization Backtest End date", value=date.today())

    # --- Optimization ---
    o1, o2 = st.columns(2, gap="small")
    with o1:
        modes = ["Max Sharpe Ratio", "Min Volatility"]
        if HAS_CVXPY:
            modes += ["Target Volatility", "Min Conditional VaR"]
        if HAS_RISKFOLIO:
            modes += ["Hierarchical Risk Parity"]
        opt_mode = st.selectbox("Optimization mode", modes)
        risk_free_rate = st.number_input("Risk-free rate (annual)", 0.0, 0.2, 0.01, 0.005, format="%.3f")

        # ⬇️ Rebalancing frequency immediately below risk-free (as requested)
        rebal_choice = st.selectbox("Rebalancing frequency", ["Monthly", "Quarterly"], index=0)
    with o2:
        max_w_asset = st.slider("Max weight per asset", 0.05, 1.0, 1.0, 0.05)
        target_vol   = st.slider("Target volatility (annualized)", 0.05, 0.40, 0.15, 0.01)
        beta_cvar    = st.slider("VaR/CVaR confidence level", 0.80, 0.99, 0.95, 0.01)

    # --- Monte Carlo & Costs (MC horizon left, Txn cost right) ---
    m1, m2 = st.columns(2, gap="small")
    with m1:
        mc_choice = st.selectbox(
            "Investment horizon",
            ["Monthly", "Quarterly", "Annual"],
            index=0
        )
        horizon = {"Monthly": 1, "Quarterly": 3, "Annual": 12}[mc_choice]
    with m2:
        txn_cost_bps = st.slider("Transaction cost (bps per rebalancing)", 0.0, 50.0, 5.0, 0.5)

    # Derived settings (unchanged)
    mc_rebal_days = 21 if rebal_choice == "Monthly" else 63
    bt_freq = "M" if rebal_choice == "Monthly" else "Q"

    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ---------- Compute on Run (sticky results) ----------
if run_btn:
    tickers = sorted(set(_current_list()))
    if not tickers:
        st.error("Your ticker list is empty.")
        st.stop()

    with st.spinner("Fetching data & computing..."):
        prices = get_price_data(tickers, start=start_date.isoformat(), end=(end_date+timedelta(days=1)).isoformat())
        if prices.empty:
            st.error("No data fetched. Check tickers or dates."); st.stop()
        rets = daily_returns(prices).replace([np.inf,-np.inf], np.nan).dropna(how="any")
        if rets.shape[0] < 30:
            st.error("Not enough clean return data after filtering."); st.stop()

        try:
            if opt_mode=="Max Sharpe Ratio":
                weights = optimize_mv(rets, "max_sharpe", rfr=risk_free_rate, max_w=max_w_asset)
            elif opt_mode=="Min Volatility":
                weights = optimize_mv(rets, "min_volatility", rfr=risk_free_rate, max_w=max_w_asset)
            elif opt_mode=="Target Volatility":
                if not HAS_CVXPY: st.error("cvxpy not installed."); st.stop()
                weights = optimize_target_vol(rets, target_vol=target_vol, max_w=max_w_asset)
            elif opt_mode=="Min Conditional VaR":
                if not HAS_CVXPY: st.error("cvxpy not installed."); st.stop()
                weights = optimize_min_cvar(rets, beta=beta_cvar, max_w=max_w_asset)
            elif opt_mode=="Hierarchical Risk Parity":
                if not HAS_RISKFOLIO: st.error("riskfolio-lib not installed."); st.stop()
                weights = optimize_hrp(rets)
        except Exception as e:
            st.error(f"Optimization failed: {e}"); st.stop()

        metrics  = port_metrics(rets, weights, rfr=risk_free_rate)
        frontier = sample_frontier(rets, n_points=50)
        vr       = hist_var_cvar(rets, weights, alpha=0.95)
        mc_stats = mc_stats_only(rets, weights, n_paths=200, horizon_days=horizon,
                                 rebalance_every=mc_rebal_days, txn_cost_bps=txn_cost_bps)
        eq_bt, summ_bt = backtest(prices, weights, freq=bt_freq, txn_cost_bps=txn_cost_bps)

    st.session_state.results = {
        "prices":prices, "frontier":frontier, "weights":weights, "metrics":metrics,
        "vr":vr, "mc_stats":mc_stats, "eq_bt":eq_bt, "summ_bt":summ_bt,
        "horizon":horizon, "rebal_choice":rebal_choice
    }
    # st.success("Done!")

# ---------- Display (sticky) ----------
res = st.session_state.results
if res is None:
    st.markdown("""
    <div style="
        display: flex;
        justify-content: center;
        margin-top: 10px;
    ">
    <div style="
        background-color: yellow;
        padding: 20px;
        font-size: 1.6rem;
        font-weight: bold;
        color: black;
        border-radius: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        text-align: left;
        white-space: nowrap;
        display: table;
    ">
        1) Enter the tickers of the stocks you're open to invest in<br> 
        2) Choose your objectives, risk appetite and investment horizon<br> 
        3) Click "Run analysis"<br>
        4) Get your customized stock portfolio !
    </div>
    </div>
    """, unsafe_allow_html=True)

else:
    prices, frontier, weights = res["prices"], res["frontier"], res["weights"]
    metrics, vr, mc_stats     = res["metrics"], res["vr"], res["mc_stats"]
    eq_bt, summ_bt            = res["eq_bt"], res["summ_bt"]
    horizon, rebal_choice     = res["horizon"], res["rebal_choice"]

    col1, col2 = st.columns([3,2], gap="large")
    with col1:

        st.markdown("""
        <div style="
            background: white;
            padding: 4px 12px;
            margin: 4px 0 12px 0;
            text-align: center;
            font-size: 1.4rem;
            font-weight: bold;
            color: black;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            border-radius: 0px;
        ">
        Starting Portfolio Backtest
        </div>
        """, unsafe_allow_html=True)
        fig = go.Figure()
        for c in prices.columns:
            fig.add_trace(go.Scatter(x=prices.index, y=prices[c], mode="lines", name=c))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price",margin=dict(t=10, b=40, l=60, r=20))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # st.markdown("""
        # <div style="
        #     background: white;
        #     padding: 4px 12px;
        #     margin: 4px 0 12px 0;
        #     text-align: center;
        #     font-size: 1.4rem;
        #     font-weight: bold;
        #     color: black;
        #     box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        #     border-radius: 0px;
        # ">
        # Efficient Frontier (50 samples)
        # </div>
        # """, unsafe_allow_html=True)

        # fig2 = go.Figure()

        # # Add frontier points
        # fig2.add_trace(
        #     go.Scatter(
        #         x=frontier["vol"] * 100,
        #         y=frontier["ret"] * 100,
        #         mode="markers",
        #         name="Frontier samples"
        #     )
        # )


        # # Configure layout
        # fig2.update_layout(
        #     xaxis_title="Volatility (annualized, %)",
        #     yaxis_title="Return (annualized, %)",
        #     margin=dict(t=10, b=40, l=60, r=20),
        #     xaxis=dict(
        #         tickformat=".0f",
        #         showgrid=True,
        #         dtick=5,
        #         zeroline=False,
        #         zerolinewidth=1,
        #         zerolinecolor="black"
        #     ),
        #     yaxis=dict(
        #         tickformat=".0f",
        #         showgrid=True,
        #         dtick=20,
        #         range=[0, max(100, (frontier["ret"].max() * 100) + 10)],  # dynamic upper range
        #         zeroline=True,
        #         zerolinewidth=1,
        #         zerolinecolor="black"
        #     ),
        #     plot_bgcolor="white"
        # )

        # st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col2:

        st.markdown("""
        <div style="
            background: white;
            padding: 4px 12px;
            margin: 4px 0 12px 0;
            text-align: center;
            font-size: 1.4rem;
            font-weight: bold;
            color: black;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            border-radius: 0px;
        ">
        Optimized Portfolio Weights
        </div>
        """, unsafe_allow_html=True)


        # --- Optimal Weights Table (fixed) ---

        # Ensure weights is defined (from optimization step)
        if isinstance(weights, pd.Series):
            weights_df = weights.reset_index()
            weights_df.columns = ['Ticker', 'Weight']
        elif isinstance(weights, dict):
            weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
        else:
            st.warning("⚠️ No weights found. Please run the analysis.")
            weights_df = pd.DataFrame(columns=['Ticker', 'Weight'])

        # Remove zero weights (optional)
        weights_df = weights_df[weights_df['Weight'] > 0]
        weights_df['Weight'] = (weights_df['Weight'] * 100).round(1).astype(str) + '%'

        # Only display if non-empty
        if not weights_df.empty:
            # Table styling
            st.markdown("""
                <style>
                .custom-table table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .custom-table th, .custom-table td {
                    text-align: center;
                    padding: 4px 8px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Render as styled HTML
            st.markdown(f"""
                <div class="custom-table" style="
                    background-color: white;
                    padding: 0;
                    border-radius: 0px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
                    font-size: 1.3rem;
                    margin-bottom: 8px;
                    max-height: 360px;
                    overflow-y: auto;
                ">
                    {weights_df.to_html(index=False, border=0)}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No optimal weights to display. Please run the analysis.")


    st.markdown("""
    <div style="
        background: white;
        padding: 4px 12px;
        margin: 4px 0 12px 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: black;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-radius: 0px;
    ">
    Efficient Frontier (50 samples)
    </div>
    """, unsafe_allow_html=True)

    fig2 = go.Figure()

    # Add frontier points
    fig2.add_trace(
        go.Scatter(
            x=frontier["vol"] * 100,
            y=frontier["ret"] * 100,
            mode="markers",
            name="Frontier samples"
        )
    )


    # Configure layout
    fig2.update_layout(
        xaxis_title="Annualized volatility in %)",
        yaxis_title="Annualized return in %)",
        margin=dict(t=10, b=40, l=60, r=20),
        xaxis=dict(
            tickformat=".0f",
            showgrid=True,
            dtick=5,
            zeroline=False,
            zerolinewidth=1,
            zerolinecolor="black"
        ),
        yaxis=dict(
            tickformat=".0f",
            showgrid=True,
            dtick=20,
            range=[0, max(100, (frontier["ret"].max() * 100) + 10)],  # dynamic upper range
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="black"
        ),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})



    st.markdown("""
    <div style="
        background: white;
        padding: 4px 12px;
        margin: 4px 0 12px 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: black;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-radius: 0px;
    ">
    Optimized Portfolio Metrics
    </div>
    """, unsafe_allow_html=True)

    fp = lambda x: f"{x:.1%}"
    left  = [("Annualized Return", fp(metrics["ann_return"])),
             ("Annualized Volatility", fp(metrics["ann_vol"])),
             ("Sharpe Ratio", f"{metrics['sharpe']:.2f}"),
             ("Daily VaR (95%)", fp(vr["VaR"])),
             ("Daily CVaR (95%)", fp(vr["CVaR"]))]

    right = [(f"Monte Carlo simulations return p. 5% ({horizon} Month)",  fp(mc_stats["p5"])),
             (f"Monte Carlo simulations return p. 50 % ({horizon} Month)", fp(mc_stats["p50"])),
             (f"Monte Carlo simulations return p. 95 % ({horizon} Month)", fp(mc_stats["p95"])),
             (f"Monte Carlo simulations average return ({horizon} Month)", fp(mc_stats["mean"])),
             (f"Monte Carlo simulations std ({horizon} Month)",  fp(mc_stats["std"])),
             (f"Backtest Total Return (Target, {rebal_choice} rebalancing)", fp(summ_bt["TotalReturn_Target"])),
             ("Backtest Total Return (Buy & Hold)", fp(summ_bt["TotalReturn_Buy&Hold"]))]

    cL, cR = st.columns(2)
    # with cL:
    #     st.dataframe(pd.DataFrame(left, columns=["Metric","Value"]), use_container_width=True, hide_index=True)
    # with cR:
    #     st.dataframe(pd.DataFrame(right, columns=["Metric","Value"]), use_container_width=True, hide_index=True)
    # with cL:
    #     table_left = pd.DataFrame(left, columns=["Metric", "Value"])
    #     st.markdown("""
    #         <div style="
    #             background-color: white;
    #             padding: 0 0 0 0;
    #             border-radius: 0px;
    #             box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    #             font-size: 1rem;
    #             margin-bottom: 8px;
    #         ">
    #         """ + table_left.to_html(index=False, border=0) + "</div>",
    #         unsafe_allow_html=True
    #     )

    with cL:
        table_left = pd.DataFrame(left, columns=["Metric", "Value"])

        # Inject custom CSS for centering + full width
        st.markdown(
            """
            <style>
            .custom-table table {
                width: 100%;
                border-collapse: collapse;
            }
            .custom-table th, .custom-table td {
                text-align: center;
                padding: 4px 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display styled table without index, centered in white box
        st.markdown(
            f"""
            <div class="custom-table" style="
                background-color: white;
                padding: 0;
                border-radius: 0px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.10);
                font-size: 1.1rem;
                margin-bottom: 8px;
            ">
                {table_left.to_html(index=False, border=0)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cR:
        table_right = pd.DataFrame(right, columns=["Metric", "Value"])

        # Inject same style if not already done
        st.markdown(
            """
            <style>
            .custom-table table {
                width: 100%;
                border-collapse: collapse;
            }
            .custom-table th, .custom-table td {
                text-align: center;
                padding: 4px 8px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display styled table without index
        st.markdown(
            f"""
            <div class="custom-table" style="
                background-color: white;
                padding: 0;
                border-radius: 0px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.10);
                font-size: 1.1rem;
                margin-bottom: 8px;
            ">
                {table_right.to_html(index=False, border=0)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style="
        background: white;
        padding: 4px 12px;
        margin: 4px 0 12px 0;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: black;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-radius: 0px;
    ">
    Backtest: Target vs Buy & Hold
    </div>
    """, unsafe_allow_html=True)

    # eq_fig = go.Figure()
    # for col in eq_bt.columns:
    #     eq_fig.add_trace(go.Scatter(x=eq_bt.index, y=eq_bt[col], mode="lines", name=col))
    # eq_fig.update_layout(
    # xaxis_title="Date",
    # yaxis_title="Performance",
    # yaxis=dict(
    #     tickformat=".0f",    # Show full numbers (no decimals)
    #     range=[0, None],     # Start Y-axis at 0
    #     showgrid=True)
    # )
    # st.plotly_chart(eq_fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Plot the Equity Curves ----
    fig = go.Figure()

    # Plot Target (Optimized with rebalancing)
    fig.add_trace(go.Scatter(
        x=eq_bt.index,
        y=eq_bt["Target"] * 100,
        mode="lines",
        name="Target (Rebalanced)"
    ))

    # Plot Buy & Hold
    fig.add_trace(go.Scatter(
        x=eq_bt.index,
        y=eq_bt["Buy & Hold"] * 100,
        mode="lines",
        name="Buy & Hold"
    ))

    y_max = math.ceil(eq_bt.max().max() * 100 / 100.0) * 100

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Equity Value Base 100",
        yaxis=dict(
            range=[0, y_max],
            tickvals=list(range(0, y_max + 100, 100)),
            tickformat=".0f",
            showgrid=True,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="black",
            title=dict(
                text="Equity value (base 100)",
                standoff=10,  # distance from axis
                font=dict(size=14)
            )
        ),
        margin=dict(t=10, b=40, l=60, r=20),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# Pinned disclaimer at the bottom of the screen
st.markdown("""
<div style='
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: black;
    color: white;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 0.9rem;
    z-index: 9999;
    text-align: center;
'>
This site does not give investment advice. Past performances don't represent future performances. <br>Data Source: Yahoo Finance.
</div>
""", unsafe_allow_html=True)
