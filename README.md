# 📊 Portfolio Optimization & Risk Dashboard

An interactive **Streamlit dashboard** for portfolio optimization, risk analysis, Monte Carlo simulations, and backtesting.  
Built with **Python, PyPortfolioOpt, Riskfolio-Lib, and Streamlit** — designed to showcase both financial knowledge and data engineering skills.

---

## 🚀 Live Demo

👉 Try it instantly on **Streamlit Cloud**:  
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://your-streamlit-app.streamlit.app)

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

### 3️⃣ Option C — Run Locally with Python

```bash
# Create and activate venv
python -m venv .venv
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
```

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

## 📈 Roadmap / Future Enhancements

- Sector & turnover constraints  
- Live data via Yahoo Finance / Tiingo APIs  
- Factor-model risk attribution  
- Kubernetes deployment (multi-service setup)  

---

## 🤝 Contributing

Pull requests welcome!  
For major changes, open an issue first to discuss what you’d like to add.  

---

## 👤 Author

**Florent Chaillie**  
Finance & Data Science professional  
- 💼 [LinkedIn](https://www.linkedin.com/in/your-linkedin/)  
- 💻 [GitHub](https://github.com/your-username)

---

## 📜 License

This project is licensed under the MIT License.
