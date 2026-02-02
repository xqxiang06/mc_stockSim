import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

"""
Use Nvidia real data to test Monte Carlo Simulation for stock GBM model
"""
nvda = pd.read_csv("data/NVIDIA_STOCK_CLEAN.csv")
# Look through last 6 months' (126 trading days) adjusted close price
prices = nvda["Adj Close"].values[-126:]
# Estimate mu and sigma from real data
# LOG Return: rt = ln( St / St-1)
log_returns = np.log(prices[1:] / prices[:-1])

mu_real = np.mean(log_returns) * 252
sigma_real = np.std(log_returns) * np.sqrt(252)

print("\n===== Parameter Estimation from NVIDIA =====")
print("Estimated mu:", mu_real)
print("Estimated sigma:", sigma_real)


"""
Monte Carlo for Stocks
"""

def mcs():
    # Parameters
    S0 = prices[-1]                # last real price as Initial
    mu = min(mu_real, 0.3)         # Expected annual return(%)
    sigma = min(sigma_real, 0.5)   # Annual volatility(%)
    T = 0.5                        # Time horizon (half year)
    n_steps = 126                  # Daily steps
    n_paths = 100000               # Number of simulations

    # model(GBM) & Formula: S(t) = S₀ × exp((μ - σ²/2)t + σW(t))
    dt = T / n_steps
    t = np.linspace(0, T, n_steps+1)

    # generate random shocks
    import time
    np.random.seed(int(time.time()))
    # For each path, generate {n_steps} random shocks
    # Each shock = random news (good or bad) affecting stock

    # dW ~ N(0, √dt)
    dW = np.random.normal( 0, np.sqrt(dt), (n_steps, n_paths) )
    
    print(f"  Generated: {n_steps} × {n_paths} = {n_steps * n_paths:,} random numbers!")
    print(f"  Sample shocks: {dW[0, :5]}")

    W = np.vstack([np.zeros(n_paths), np.cumsum(dW, axis=0)])
    print(f"  W shape: {W.shape} = ({n_steps+1} time points × {n_paths} paths)")

    # for each path, for each time point, calculate price
    time_grid = t.reshape(-1, 1) # transform an array with a single column and multiple rows (a column vector)
    
    S = S0 * np.exp( (mu - 0.5 * sigma**2) * time_grid + sigma * W )
    
    print(f"  Initial prices: ${S[0, :5]}")
    print(f"  Final prices (sample): ${S[-1, :5]}")

    # retrieve all paths
    final_prices = S[-1, :]
    return t, S, S0, T

t, S, S0, T = mcs() # function call
   

# graphics
def plot_paths_terminal_payoff_dist(t, S, S0):
    # Visualization for Monte Carlo stock paths and terminal price distribution
    plt.figure(figsize=(20, 5))

    # ---- (1) Stock price paths ----
    plt.subplot(1, 2, 1)
    plt.plot(t, S[:, :10], linewidth=1.1, alpha=0.8)  # plot only 10 paths
    plt.axhline(y=S0, linestyle="--", label="Initial Price")
    plt.title("Monte Carlo Simulated Stock Prices (GBM)")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)

    # ---- (2) Terminal price distribution ----
    plt.subplot(1, 2, 2)
    plt.hist(S[-1, :], bins=60, density=True)
    plt.axvline(np.mean(S[-1, :]), linestyle="--", label="Mean $S_T$")
    plt.title("Distribution of Terminal Stock Prices $S_T$")
    plt.xlabel("Terminal Stock Price")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_paths_terminal_payoff_dist(t, S, S0) # function call


# plot: real nvidia price vs. monte carlo simulaion
def plot_real_vs_mc(nvda_prices, t, S):
    # Real NVIDIA price
    t_hist = np.linspace(-0.5, 0, len(nvda_prices)) # Map to the negative timeline (past)
    plt.plot(t_hist, nvda_prices, label="Real NVIDIA (Past half year)", color = 'pink')
    # Add the "Today" dividing line
    plt.axvline(0, color='red', linestyle='--', label='TODAY')
    
    # -----Monte Carlo median path - select the path closest to real's median-----
    # Median at each time point
    median_path = np.median(S, axis=1) 
    # Distance between each path and the median path
    distances = np.mean(np.abs(S - median_path.reshape(-1, 1)), axis=0)
    # returns index of first occurrence of min value (distance)
    best_idx = np.argmin(distances)

    plt.plot(t, S[:, best_idx], linestyle='--', 
             label=f"MC path_mae", alpha=0.8)

    plt.xlim(-0.5, 0.5)
    plt.title("Real NVIDIA vs Monte Carlo GBM")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_real_vs_mc(prices, t, S)


# check confidence interval
def compute_confidence_interval(S, confidence=0.95):
    # S: simulated stock price paths (n_steps+1, n_paths)
    # confidence: confidence level (default 95%)
    lower_p = (1 - confidence) / 2
    upper_p = 1 - lower_p

    # Take quantiles by column (along each row)
    lower_band = np.percentile(S, 100*lower_p, axis=1)
    upper_band = np.percentile(S, 100*upper_p, axis=1)
    mean_path  = np.mean(S, axis=1)

    return lower_band, upper_band, mean_path

# plot
def plot_mc_with_confidence(t, S, S0):
    lower, upper, mean_path = compute_confidence_interval(S)

    plt.figure(figsize=(8,5))
    # draw the path for mean
    plt.plot(t, mean_path, label="MC Mean Prediction", color="orange", linewidth=2)
    # graph he confidence interval band
    plt.fill_between(t, lower, upper, color="blue", alpha=0.2, label="95% Confidence Interval")
    # the initial price level line
    plt.axhline(y=S0, linestyle="--", color="gray", label="Initial Price")

    plt.title("Monte Carlo Prediction with 95% Confidence Interval")
    plt.xlabel("Time (years)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_mc_with_confidence(t, S, S0) # function call

