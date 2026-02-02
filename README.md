# mc_stock_sim
A C++ stock price simulator based on stochastic processes (Geometric Brownian motion). Historical market data is loaded from CSV files to estimate drift and volatility, and simulation results are exported for further analysis and visualisation.

---

## Models

### 1. Geometric Brownian Motion (GBM)

The baseline model assumes continuous, log-normally distributed price changes with constant drift and volatility.

```
dS = μS dt + σS dW
```

In log-space (used for simulation stability):

```
d(ln S) = (μ − σ²/2) dt + σ dW
```

`μ` and `σ` are estimated from historical daily log returns and annualized using 252 trading days.

#### Simulation Results
   
![Real NVIDIA vs Monte Carlo GBM](pic/simulate_path.png)
   
*Figure 1: Comparison of real NVIDIA stock prices (past 6 months) vs Monte Carlo GBM simulation*

---

### 2. Jump Diffusion (Merton Model)

Extends GBM with a compound Poisson process to capture sudden, large price moves that pure diffusion cannot explain — earnings surprises, flash crashes, or macro shocks.

```
d(ln S) = (μ − σ²/2 − λκ) dt + σ dW + J dN
```

| Component | Description |
|-----------|-------------|
| `N ~ Poisson(λ)` | Number of jumps per unit time |
| `J ~ N(μ_J, σ_J²)` | Log-size of each jump |
| `κ = E[eʲ − 1]` | Expected percentage price change per jump; used to correct the drift so the mean path is preserved |
| `σ_smooth` | Volatility estimated from non-jump returns only |

**Calibration** separates jumps from normal moves using a Z-score threshold on daily log returns. Days exceeding the threshold (default 2.5σ) are classified as jumps; the remaining days contribute to `σ_smooth`. Jump frequency `λ`, mean size `μ_J`, and jump volatility `σ_J` are estimated directly from the classified jump returns.

---

### 3. Regime-Switching (Hamilton Model)

Models the market as alternating between two latent states — **Normal** and **Crash** — each with its own drift and volatility, governed by a first-order Markov chain.

```
Z_t ∈ {Normal, Crash}       (hidden regime)

dS/S | Z_t = i  ~  GBM(μ_i, σ_i)

Transition matrix P:
    ┌                        ┐
P = │  1 − p_NC      p_NC    │   (from Normal)
    │  p_CN      1 − p_CN    │   (from Crash)
    └                        ┘
```

**Stationary distribution** (long-run fraction of time in each regime):

```
π_Normal = p_CN / (p_NC + p_CN)
π_Crash  = p_NC / (p_NC + p_CN)
```

**Calibration pipeline:**

1. Compute daily log returns from historical prices.
2. Compute a rolling-window annualized volatility series.
3. Run K-means clustering (k = 2) on the volatility series to assign each day to Normal or Crash.
4. Estimate `μ_i` and `σ_i` per regime from the classified returns.
5. Count transitions in the regime sequence to estimate `p_NC` and `p_CN`.

A longer lookback window (default 252 days) is used for regime calibration compared to jump/GBM estimation (126 days), giving the clustering algorithm better separation between the two volatility clusters.

---

## Option Pricing

European call and put options are priced under the **risk-neutral measure** using Monte Carlo simulation and compared against analytical Black-Scholes prices.

Under the risk-neutral measure, `μ` is replaced by the risk-free rate `r`, and the drift compensation for jumps is applied identically:

```
d(ln S) = (r − σ²/2 − λκ) dt + σ dW + J dN
```

The option value is the discounted expected payoff:

```
V₀ = e^(−rT) · E^Q[ payoff(S_T) ]
```

The **jump premium** — the percentage by which the jump-diffusion price exceeds the Black-Scholes price — quantifies the market's compensation for tail risk that a pure-diffusion model ignores.

---

## Prerequisites

- C++17 compiler (GCC 11+ or Clang 14+)
- Intel TBB (`std::execution::par` support)
  - macOS: `brew install tbb`
  - Linux: `sudo apt install libtbb-dev`
- Python 3.7+ with `yfinance` — for fetching market data
  - `pip install yfinance`

---

## Clone

```bash
git clone https://github.com/xqxiang06/mc_stockSim.git
cd mc_stockSim
```

---

## Get Data

Download historical price data for any ticker before running the simulator:

```bash
# NVIDIA — last 2 years (default)
python fetch_data.py NVDA

# Any ticker, custom date range
python fetch_data.py AAPL -s 2022-01-01 -e 2024-12-31

# Maximum available history
python fetch_data.py TSLA -p max

# Custom output path
python fetch_data.py NVDA -p 5y -o data/nvda_stock.csv
```

Output lands in `data/<TICKER>_STOCK_DATA.csv` by default. If you change the ticker or filename, update `csv_file` at the top of `main.cpp` to match.

---

## Build & Run

```bash
make            # compile → bin/monte_carlo_sim
make run        # build (if needed) + run
make debug      # rebuild with -g -O0 for debugging
make clean      # remove object files and binary
make clean-all  # also remove generated CSVs in data/
```

Override the compiler if needed:

```bash
make CXX=clang++
```

---

## Configuration

All user-facing parameters live at the top of `main()` in `main.cpp`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_file` | `data/nvda_stock.csv` | Path to historical price data |
| `lookback_days` | 126 | Trading days used for GBM/jump calibration |
| `regime_lookback_days` | 252 | Trading days used for regime calibration |
| `n_paths` | 1,000,000 | Number of Monte Carlo paths |
| `n_steps` | 126 | Time steps per path (daily for 6 months) |
| `T` | 0.5 | Simulation horizon in years |
| `jump_threshold` | 2.5 | Z-score cutoff for classifying jumps |
| `r` | 0.045 | Risk-free rate for option pricing |

---

## Output Summary

A typical run prints calibrated parameters for all three models, per-model statistics (mean, median, 95% confidence interval, expected return), regime transition matrices and stationary distributions, and option prices with jump premiums. All numerical results are also written to the CSV files listed in the project structure above.