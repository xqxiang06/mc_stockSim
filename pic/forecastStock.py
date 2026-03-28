# Stock Forecast Chart, AAPL as an example
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Config
HIST_CSV   = "data/AAPL_STOCK_DATA.csv" # historial prices
PATHS_CSV  = "data/mc_results_gbm_paths.csv" # C++ results
OUTPUT_PNG = "pic/stockForecast.png"
LOOKBACK   = 252 # trading days of history to show

# Load historial prices
hist_df = pd.read_csv(HIST_CSV)
hist_df['Date'] = pd.to_datetime(hist_df['Date'])
hist_df = hist_df.sort_values('Date').tail(LOOKBACK)
hist_prices = hist_df['Adj Close'].to_numpy()
hist_days = np.arange(-len(hist_prices), 0) # -252 .. -1
 
S0 = hist_prices[-1] # last observed price = simulation start

# Load MC paths
paths_df = pd.read_csv(PATHS_CSV)
# Raw data to pivot table
pivot = paths_df.pivot(index="step", columns="path_id", values="price")
 
sim_steps  = pivot.index.to_numpy()
# median across all paths at each step
sim_prices = pivot.median(axis=1).to_numpy()

# Plot
fig, ax = plt.subplots(figsize=(12,6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

TEXT  = "#c8d0e0"
GRID  = "#1e2533"
PINK  = "#f4a7b9"
BLUE  = "#0abab5"
RED   = "#e05c5c"

# Historical line (left)
ax.plot(hist_days, hist_prices, color=PINK,
        linewidth=1.5, alpha=0.9, label="AAPL (past year)")
# "Today" vertical divider
ax.axvline(0, color=RED, linewidth=1.4, linestyle="--", label="TODAY")
# Median forecast path (right)
ax.plot(sim_steps, sim_prices, color=BLUE,
        linewidth=1.5, alpha=0.9, label="GBM median path")
# Dot at S0 junction
ax.scatter([0], [S0], color=RED, zorder=5, s=40)
# Annotate final forecast price
final_price = sim_prices[-1]
ax.annotate(f"${final_price:.2f}",
            xy=(sim_steps[-1], final_price),
            xytext=(sim_steps[-1] - 20, final_price + (S0 * 0.03)),
            color=TEXT, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.8))

# Styling
ax.set_xlabel("Trading day (0 = today)", color=TEXT, fontsize=10)
ax.set_ylabel("Stock price ($)", color=TEXT, fontsize=10)
ax.set_title("AAPL — Historical vs GBM Monte Carlo Forecast",
             color=TEXT, fontsize=13, pad=14)
 
ax.tick_params(colors=TEXT)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
ax.grid(True, color=GRID, linewidth=0.6)
 
ax.legend(facecolor="#1a1f2e", edgecolor=GRID,
          labelcolor=TEXT, fontsize=9, loc="upper left")
 
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUTPUT_PNG}")
plt.show()