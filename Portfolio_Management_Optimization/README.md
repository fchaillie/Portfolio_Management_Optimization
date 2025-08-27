# Portfolio Optimization Dashboard

A client‑facing web application that supports the **design, analysis, and backtesting** of long‑only equity portfolios. The interface is built for clarity and auditability, while the engine applies established quantitative techniques (mean–variance, target volatility, CVaR, and HRP) with transparent assumptions and reproducible results.

> **Important**  
> This software is provided for research and educational purposes. It is **not** investment advice and should not be used to make investment decisions. Past performance is not indicative of future results.

---

## 1. Executive Summary

- **Objective:** Provide a disciplined workflow to construct and compare long‑only portfolios under different risk frameworks.  
- **Audience:** Investment professionals, analysts, and sophisticated clients seeking transparent methodology and quick iteration.  
- **What you can do:**  
  - Specify an investable universe and date range.  
  - Optimize using Max Sharpe, Min Volatility, **Target Volatility**, **Min CVaR**, or **HRP**.  
  - Review risk/return metrics (annualized), VaR/CVaR, and bootstrapped Monte Carlo horizon outcomes.  
  - Backtest an optimized portfolio against buy‑and‑hold, including rebalancing costs.

---

## 2. Capabilities

- **Optimization methods**
  - **Max Sharpe / Min Volatility** (PyPortfolioOpt with Ledoit–Wolf shrinkage and L2 regularization)
  - **Target Volatility** (efficient frontier point at a chosen annualized volatility)
  - **Min Conditional VaR (β)** (requires `cvxpy`; minimizes tail loss at confidence β)
  - **Hierarchical Risk Parity (HRP)** (via `riskfolio-lib`)

- **Risk & performance analytics**
  - Annualized return, volatility, Sharpe ratio
  - **Historical VaR/CVaR** (non‑parametric) on daily portfolio P&L
  - **Monte Carlo (bootstrap)** of daily returns for 1/3/12‑month horizons, deterministic seed

- **Backtesting**
  - Monthly/Quarterly rebalancing
  - Simple transaction costs in basis points applied to turnover
  - Equity curve comparison vs. buy‑and‑hold, total return summary

- **User experience**
  - Single‑click “Run analysis” with **sticky results** (no partial live recompute)
  - Wide sidebar for efficient data entry
  - Clear, print‑ready tables and charts

---

## 3. Methodology (at a glance)

- **Returns & Covariance:** Daily log‑approx returns from adjusted close. Covariance estimated via **Ledoit–Wolf shrinkage**.  
- **Mean Estimate:** Historical mean return (compounded) on the same daily series.  
- **Regularization:** Small **L2 penalty** added in mean–variance programs to stabilize weights.  
- **Annualization:** 252 trading days.  
- **Risk Measures:**  
  - **VaR/CVaR:** Historical (empirical) distribution of daily portfolio returns.  
  - **Monte Carlo:** Non‑parametric resampling (bootstrap) of daily return vectors with optional periodic rebalancing and turnover costs.  
- **Backtest:** Target weights drift between rebalances; turnover cost applied at each rebalance; baseline is buy‑and‑hold from initial weights.

**Limitations:** Historical estimates assume stationarity and i.i.d. daily draws in the bootstrap; regime shifts, liquidity, market impact, and tax frictions are not modeled.

---

## 4. Data & Coverage

- **Source:** Yahoo Finance via `yfinance` (adjusted close).  
- **Frequency:** Daily (business days).  
- **Coverage:** Dependent on ticker availability and corporate actions. Missing data are forward/back‑filled over short gaps only.  
- **Quality Note:** Public data may contain anomalies; users should validate critical inputs independently.

---

## 5. Controls & Governance

- **Determinism:** Monte Carlo uses a fixed seed for repeatability.  
- **Transparency:** Inputs, parameters, and chosen optimizer are visible in the UI.  
- **Bounded Weights:** Long‑only with configurable **max weight per asset**.  
- **Optionality Flags:** UI enables CVaR/HRP only if dependencies (`cvxpy`, `riskfolio-lib`) are present.  
- **Auditability:** The codebase is modular (`src/portfolio_app`) and can be reviewed or extended; results are reproducible with the same inputs.

---

## 6. Installation & Launch

### Python Environment
```bash
# Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application
```bash
streamlit run app/main.py
```
(Optional) Place a custom background at `assets/background.jpg`.

---

## 7. Using the Application

1. **Universe:** Paste or add tickers in the sidebar (one per line).  
2. **Dates:** Select start and end dates for data and backtesting.  
3. **Optimization:** Choose method, risk‑free rate, and (where applicable) target volatility or CVaR β.  
4. **Constraints:** Set maximum per‑asset weight.  
5. **Rebalancing & Costs:** Choose Monthly/Quarterly; specify transaction costs in bps.  
6. **Run:** Click **Run analysis**. Results remain fixed until the next run.  
7. **Review Outputs:**  
   - Price history for the starting universe  
   - **Optimized Weights** (sorted by descending weight)  
   - **Efficient Frontier** samples  
   - **Metrics:** Annualized return/vol, Sharpe, VaR/CVaR, Monte Carlo  
   - **Backtest:** Target vs buy‑and‑hold equity curves and total returns

---

## 8. Configuration Notes

- **Max weight per asset:** 5–100% slider; enforces diversification.  
- **Risk‑free rate:** Used in Sharpe maximization and metrics.  
- **CVaR β:** Tail confidence (e.g., 0.95); lower β increases tail sensitivity.  
- **Monte Carlo horizon:** Select 1/3/12 months; rebalancing interval set by Monthly/Quarterly selection.  
- **Transaction costs:** Applied to turnover at each rebalance in bps (1 bps = 0.01%).

---

## 9. Extensibility Roadmap

- **Additional constraints:** Sector caps, minimum weights, exclusion lists.  
- **Factor views:** Black–Litterman priors; robust covariance estimators.  
- **Multi‑asset support:** Bonds, commodities, FX; multi‑currency handling and FX hedging.  
- **Reporting:** Exportable PDF/CSV reports; scenario save/load.  
- **Deployment:** Containerization, CI/CD, and cloud hosting with authentication.  
- **Testing:** Unit tests for data transforms, optimizers, and backtesting logic.

---

## 10. Security & Privacy

- The app does not collect personal data by default.  
- If deployed in a shared environment, ensure appropriate **authentication**, **network controls**, and **logging**.  
- Review third‑party data terms and conditions before production use.

---

## 11. Compliance & Disclaimers

- This tool is not a portfolio management service and does not provide investment advice or solicitations.  
- Outputs are model‑based estimates subject to input quality and modeling assumptions.  
- Users are responsible for independent due diligence and regulatory compliance in their jurisdiction.

---

## 12. Technical Stack

- **UI:** Streamlit, Plotly  
- **Quant:** PyPortfolioOpt, NumPy, Pandas  
- **Data:** `yfinance` (Yahoo Finance adjusted close)  
- **Optional:** `cvxpy` (CVaR), `riskfolio-lib` (HRP)

---

## 13. Repository Layout

```
.
├─ app/
│  └─ main.py                 # Streamlit entry point
├─ src/
│  └─ portfolio_app/
│     ├─ __init__.py          # package metadata
│     ├─ ui.py                # CSS, banner, disclaimer, background, sidebar width
│     ├─ data.py              # price download + daily returns
│     ├─ optimizers.py        # MV, Target-Vol, Min-CVaR, HRP (+ capability flags)
│     ├─ metrics.py           # frontier sampling, metrics, VaR/CVaR, Monte Carlo
│     └─ backtest.py          # rebalance vs buy & hold, turnover cost
├─ assets/
│  └─ background.jpg          # optional
├─ requirements.txt
└─ README.md
```

---

## 14. Contact

For client demonstrations, integration discussions, or a technical walkthrough, please reach out to the project maintainer.
