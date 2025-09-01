# 📊 Portfolio Optimization & Risk Dashboard

Here is my interactive dashboard for portfolio optimization, risk analysis, Monte Carlo simulations and backtesting.  
Built with Python and Streamlit, this project brings **quantitative finance** concepts to life letting users explore and compare investment strategies in real time.
Whether you're a trader, an investor or a recruiter curious about my technical & financial skills, this dashboard shows how I combine **data science, portfolio theory and modern deployment practices** to build tools that solve real-world problems.

---

## 🌐 [Live Demo](https://portfolio-optimizer-fchaillie.fly.dev)

![CI/CD Pipeline](https://github.com/fchaillie/Portfolio_Management_Optimization/actions/workflows/ci-cd.yaml/badge.svg)

---

### 🌟 What makes it special?
- 📊 **Dynamic Portfolio Optimization** — Mean-Variance, CVaR and Target Volatility models at your fingertips.  
- 🎲 **Monte Carlo Simulations** — Explore thousands of portfolio scenarios instantly.  
- 🔄 **Backtesting Engine** — See how strategies would have performed historically.  
- 🎨 **Polished UI** — Simple graphs and tables making finance insights easy to understand.  
- ☁️ **Easy Access** — Try it online via Streamlit Cloud or deploy it anywhere via Docker in just one command.  

---

## 🖼️ Screenshots
 
### Dashboard home

![alt text](assets/image-1.png)

### Equally weighted chosen portfolio backtest performance

![alt text](assets/image-2.png)

### Efficient Frontier

![alt text](assets/image-4.png)

### Optimized portfolio backtest performance 

![alt text](assets/image-5.png)

### Optimized portfolio backtest and forward looking (Monte Carlo) metrics 

![alt text](assets/image-6.png)


---

## ⚡ Quick Start

### 1️⃣ Option A — Run Online
Fastest way: click the link at the top ☝️  
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

This project includes unit tests to ensure correctness and reliability:

- **Data validation**: checks that price data is returned as a non-empty Pandas DataFrame with all requested tickers.  
- **Metrics**: verifies daily returns and portfolio performance metrics are calculated correctly.  
- **Optimizers**: ensures portfolio optimizers (e.g., Max Sharpe, Min Volatility) return valid weights that sum to 1.  

Tests are run locally with **pytest** and automatically in CI/CD via **GitHub Actions**, guaranteeing stable builds and reproducible results.

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

```bash
Portfolio_Management_Optimization/
├── app/ # Streamlit app (UI)
├── src/portfolio_app/ # Core logic (data, optimizers, backtest, metrics)
├── tests/ # Unit tests
├── assets/ # Images for README and dashboard deployed
├── requirements.txt # Python dependencies
├── Dockerfile # Deployment config
└── README.md # Documentation
```

---

## 📜 License

This project is licensed under the MIT License.
