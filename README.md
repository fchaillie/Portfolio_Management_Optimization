# 📊 Portfolio Optimization & Risk Dashboard

Here is my interactive dashboard for portfolio optimization, risk analysis, Monte Carlo simulations and backtesting.  
Built with Python and Streamlit, this project brings **quantitative finance** concepts to life letting users explore and compare investment strategies in real time.
Whether you're a trader, an investor or a recruiter curious about my technical & financial skills, this dashboard shows how I combine **data science, portfolio theory, and modern deployment practices** to build tools that solve real-world problems.

---

## 🌐 Live Demo
👉 (https://portfolio-optimizer-fchaillie.fly.dev)
#[Always-On App (Fly.io)]
---

![CI/CD Pipeline](https://github.com/fchaillie/Portfolio_Management_Optimization/actions/workflows/ci-cd.yaml/badge.svg)

---

## ✨ Features

- Multiple optimization methods:
  - **Mean-Variance** (max Sharpe, min volatility, target volatility)
  - **CVaR (Conditional Value-at-Risk)**
  - **HRP (Hierarchical Risk Parity)**
- **Backtesting engine** with:
  - Rolling window optimization
  - Transaction costs & slippage modeling
  - Benchmark comparison (SPY, equal-weight, 60/40)
- **Monte Carlo simulations** (parametric & bootstrap) for forward-looking risk analysis
- Interactive **visualizations** with Plotly & Matplotlib
- Export results to **CSV (weights, metrics)** and **PNG (charts)**
- Custom **UI styling** (CSS + Streamlit components)

---

## 🖼️ Screenshots

📌 Suggested places to add visuals:  
- Dashboard home (screenshot)  
- Efficient Frontier example (screenshot)  
- Monte Carlo simulation (screenshot)  
- Backtest performance curve (screenshot)  
- Short GIF demo of a full run  

```markdown
![Dashboard Home](images/dashboard_home.png)
![Efficient Frontier](images/efficient_frontier.png)
![Monte Carlo](images/montecarlo.png)
![Backtest](images/backtest.png)
```

💡 Tip: include a short **GIF demo** (`images/demo.gif`) showing someone selecting tickers, running optimization, and viewing results.

---

## ⚡ Quick Start

### 1️⃣ Option A — Run Online (Streamlit Cloud)
Fastest way: click the badge above ☝️  
No installation required.

---

### 2️⃣ Option B — Run Locally with Docker

Clone the repo and build the container:

```bash
git clone https://github.com/<your-username>/portfolio-optimizer.git
cd portfolio-optimizer
docker build -t portfolio-opt .
docker run -p 8501:8501 portfolio-opt
```

Then open [http://localhost:8501](http://localhost:8501) 🎉

---

## 🧪 Testing

Run the unit tests with:

```bash
pytest -v
```

These tests validate:  
- Optimizers output valid weight vectors  
- Backtest logic runs on toy data  
- Streamlit app loads without errors  

CI (GitHub Actions) automatically runs tests on every push.

---

## 🛠️ Tech Stack

- **Python** (3.11)  
- **Streamlit** (UI)  
- **PyPortfolioOpt** (classical optimizers)  
- **Riskfolio-Lib** (CVaR, HRP)  
- **Pandas / NumPy / SciPy**  
- **Plotly / Matplotlib**  
- **Docker** (deployment)  
- **GitHub Actions** (CI)  

---

## 📂 Project Structure

```
├── app.py                # Streamlit app entry point
├── optimizers.py         # Mean-Variance, HRP, CVaR functions
├── backtest.py           # Rolling window backtest engine
├── montecarlo.py         # Monte Carlo simulations
├── metrics.py            # Risk & performance metrics
├── tests/                # Unit tests (pytest)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License.
