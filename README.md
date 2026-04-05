# 📈 Market Regime Detection Engine

An academic-grade tool for identifying hidden market states (regimes) using statistical modeling. This engine leverages **Hidden Markov Models (HMM)** and **K-Means clustering** to partition financial time series data into distinct volatility and return environments.

## 🎯 Project Overview

The **Market Regime Detection Engine** is designed for quantitative analysts and traders who need to understand the underlying "mood" of a market. By analyzing historical log returns and rolling volatility, the engine can distinguish between different market states, such as:
- 🟢 **Quiet/Growth:** Low volatility with steady positive returns.
- 🔴 **Turbulent/Correction:** High volatility with negative or erratic returns.
- 🟡 **High Volatility/Rally:** Significant price swings but with an upward bias.

### ✨ Key Features
- 🤖 **HMM (Hidden Markov Model):** A probabilistic approach that assumes market regimes are hidden states of a Markov process. It computes transition probabilities (the likelihood of moving from one state to another).
- 🧩 **K-Means Clustering:** A distance-based approach to group data points with similar return/volatility profiles.
- 🎲 **Synthetic Data Generation:** Includes a Geometric Brownian Motion (GBM) generator for testing models in controlled environments.
- 📊 **Visual Analytics:** Generates price-regime overlays, state probability charts, and transition matrix heatmaps.

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.10+
- Dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`, `hmmlearn`, `scikit-learn`

### 🔧 Installation
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

The engine is a CLI-based application. You can provide your own CSV data or generate synthetic data.

### 1. 📂 Using Real Data (CSV)
The engine expects a CSV with at least `date` and `close` columns.
```bash
python market_regime_detection_engine.py --csv path/to/your_data.csv --n_states 3
```

### 2. 🧪 Using Synthetic Data
Generate a Geometric Brownian Motion series to test the model:
```bash
python market_regime_detection_engine.py --gbm --n_states 3
```

### ⚙️ CLI Arguments
- `--csv`: Path to a CSV file (must contain `date` and `close`).
- `--gbm`: Use synthetic data instead of a file.
- `--n_states`: Number of regimes to detect (default: 3).
- `--window`: Rolling window size for feature engineering (default: 20).
- `--model`: Model type to use (`HMM` or `KMeans`).

## 📤 Outputs
- 📉 **Transition Matrix:** (HMM only) Shows the probability of switching between regimes.
- 🧾 **Regime Statistics:** Summary table showing Mean Return, Volatility, and Count for each detected state.
- 🖼️ **Visualizations:**
  - Price chart colored by detected regime.
  - Probability of being in each regime over time.
  - Heatmap of the transition matrix.

## ⚖️ License
MIT License
