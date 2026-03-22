# Three-fund Portfolio graphs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # 2D geometric shapes

# Config
PATHS_CSV = "data/portfolio_results_paths.csv"
OUTPUT_PNG = "pic/fanChart.png"
INITIAL_VAL = 10000.0

df = pd.read_csv(PATHS_CSV)

# Pivot: Rows = time, Columns = different simulations
pivot = df.pivot(index = "step", columns = "path_id", values = "portfolio_value")
steps = pivot.index.to_numpy() # convert index to a NumPy array: shape (253, )
trading_days = steps # x-axis in days

# Compute percentiles at each step
pcts = {p: pivot.quantile(p/100, axis = 1).to_numpy()
        for p in [10, 25, 50, 75, 90]}

# Plot in Dark Theme
fig, ax = plt.subplots(figsize = (11,6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

BLUE_DARK  = "#1a3a5c"; 
BLUE_MID   = "#1e5fa8"
BLUE_LINE  = "#4a9eff"
RED_DASHED = "#e05c5c"
GRID_COLOR = "#1e2533"
TEXT_COLOR = "#c8d0e0"

# Outer band: 10th - 90th
ax.fill_between(trading_days, pcts[10], pcts[90],
                color=BLUE_DARK, alpha=0.6, linewidth=0)
# Inner band: 25th - 75th
ax.fill_between(trading_days, pcts[25], pcts[75],
                color=BLUE_MID, alpha=0.7, linewidth=0)
# Median line
ax.plot(trading_days, pcts[50], color = BLUE_LINE, linewidth = 2.2,
        label = "Median", zorder = 3)
# Initial value baseline
ax.axhline(INITIAL_VAL, color = RED_DASHED, linewidth = 1.3,
           linestyle = "--", label = f"Initial ${INITIAL_VAL:,.0f}", zorder = 2)

# Annotations at final step
final = trading_days[-1]
label_cfg = dict(fontsize=8.5, color=TEXT_COLOR,
                 va="center", ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#0f1117", ec="none", alpha=0.7))

for pct, key, label in [
    (pcts[90][-1], 90, "90th"),
    (pcts[75][-1], 75, "75th"),
    (pcts[50][-1], 50, "50th (median)"),
    (pcts[25][-1], 25, "25th"),
    (pcts[10][-1], 10, "10th"),
]:
    ax.annotate(f"${pct:,.0f}  {label}",
                xy=(final, pct), xytext=(final + 4, pct),
                **label_cfg)
    
# Styling
ax.set_xlim(0, trading_days[-1] + 40) # right margin for labels
ax.set_xlabel("Trading day", color=TEXT_COLOR, fontsize=10)
ax.set_ylabel("Portfolio value ($)", color=TEXT_COLOR, fontsize=10)
ax.set_title("Three-Fund Portfolio — Monte Carlo Fan Chart",
             color=TEXT_COLOR, fontsize=13, pad=14)

ax.tick_params(colors=TEXT_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COLOR)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="-")

# Legend
band_outer = mpatches.Patch(color=BLUE_DARK,  alpha=0.8, label="10th–90th pct")
band_inner = mpatches.Patch(color=BLUE_MID,   alpha=0.9, label="25th–75th pct")
median_ln  = plt.Line2D([0], [0], color=BLUE_LINE,  linewidth=2, label="Median")
base_ln    = plt.Line2D([0], [0], color=RED_DASHED, linewidth=1.3,
                        linestyle="--", label=f"Initial ${INITIAL_VAL:,.0f}")
 
ax.legend(handles=[band_outer, band_inner, median_ln, base_ln],
          facecolor="#1a1f2e", edgecolor=GRID_COLOR,
          labelcolor=TEXT_COLOR, fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi = 150, bbox_inches = "tight", facecolor = fig.get_facecolor())
print(f"Saved → {OUTPUT_PNG}")
plt.show()