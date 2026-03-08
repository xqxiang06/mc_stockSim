"""
C++ Monte Carlo Simulation Results Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory for plots
output_dir = Path('pic')
output_dir.mkdir(exist_ok=True)

# ==================== Load Data ====================
try:
    # Summary statistics
    summary = pd.read_csv('data/mc_summary.csv')
    
    # Individual model results
    gbm_results = pd.read_csv('data/mc_results_gbm.csv')
    jump_results = pd.read_csv('data/mc_results_jump.csv')
    regime_results = pd.read_csv('data/mc_results_regime.csv')
    
    # Portfolio results
    portfolio_results = pd.read_csv('data/portfolio_results.csv')
    
    # Option prices
    options = pd.read_csv('data/option_prices.csv')
    
    print("All data loaded successfully.\n")
    
except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}")
    exit(1)

stock_data = pd.read_csv("data/AAPL_STOCK_DATA.csv")
prices = stock_data["Adj Close"].values[-126:]
S0 = prices[-1]  # Initial stock price (use the last day price)

PORTFOLIO_INITIAL = 10000.0  # Initial portfolio value

# ==================== 1. Stock Price Distribution Comparison ====================

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle('Simulation of AAPL Final Stock Price Distributions (Per Share)', fontsize=16)

# GBM
ax1 = axes[0, 0]
ax1.hist(gbm_results['final_price'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(summary[summary['model'] == 'GBM']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax1.axvline(S0, color='gray', linestyle=':', linewidth=2, label=f'Initial: ${S0:.2f}')
ax1.set_title('Pure Geometric Brownian Motion')
ax1.set_xlabel('Final Price ($)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Jump Diffusion
ax2 = axes[0, 1]
ax2.hist(jump_results['final_price'], bins=50, alpha=0.7, color='coral', edgecolor='black')
ax2.axvline(summary[summary['model'] == 'Jump']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax2.set_title('Jump Diffusion (Merton Model)')
ax2.set_xlabel('Final Price ($)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Regime-Switching
ax3 = axes[1, 0]
ax3.hist(regime_results['final_price'], bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
ax3.axvline(summary[summary['model'] == 'Regime']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax3.set_title('Regime-Switching (Normal-Crash)')
ax3.set_xlabel('Final Price ($)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Boxplot comparison
ax4 = axes[1, 1]
# Data for box plot
data_to_plot = [
    gbm_results['final_price'],
    jump_results['final_price'],
    regime_results['final_price']
]

bp = ax4.boxplot(data_to_plot, 
                tick_labels=['GBM', 'Jump Diffusion', 'Regime-Switching'],
                patch_artist=True,  # Fill boxes with color (allowing for custom color schemes)
                notch=True,  # Create a notched box plot (notches represent the CI around median)
                showmeans=True)

# Color the boxes
colors = ['steelblue', 'coral', 'mediumseagreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean values as text
for i, model in enumerate(['GBM', 'Jump', 'Regime']):
    mean_val = summary[summary['model'] == model]['mean_price'].values[0]
    ax4.text(i+1, mean_val, f'${mean_val:.2f}', 
            ha='center', va='bottom')

ax4.axhline(S0, color='gray', linestyle=':', linewidth=2, label=f'Initial: ${S0:.2f}')
ax4.legend()
ax4.set_title('All Stock Model Boxplot Comparison')
ax4.set_ylabel('Final Stock Price ($)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'StockModelCompar.png', dpi=300, bbox_inches='tight')
print(f"    Saved: pic/StockModelCompar.png")
plt.show()

# ==================== PLOT 2: Portfolio Analysis ====================

# Analyze portfolio separately with components
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle('Portfolio Analysis (60/40 AAPLstock/Bond) - $10,000 Investment', fontsize=16)

# Portfolio value distribution
ax1 = axes[0, 0]
ax1.hist(portfolio_results['portfolio_value'], bins=50, alpha=0.7, 
            color='purple', edgecolor='black')
mean_pf = portfolio_results['portfolio_value'].mean()
ax1.axvline(mean_pf, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: ${mean_pf:.2f}')
ax1.axvline(PORTFOLIO_INITIAL, color='gray', linestyle=':', linewidth=2, 
            label=f'Initial: ${PORTFOLIO_INITIAL:.2f}')
ax1.set_title('Portfolio Value Distribution')
ax1.set_xlabel('Final Portfolio Value ($)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Stock component
ax2 = axes[0, 1]
ax2.hist(portfolio_results['stock_price'], bins=50, alpha=0.7, 
            color='steelblue', edgecolor='black')
mean_stock = portfolio_results['stock_price'].mean()
ax2.axvline(mean_stock, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: ${mean_stock:.2f}')
ax2.axvline(S0, color='gray', linestyle='--', label=f'Initial: ${S0:.2f}')
ax2.set_title('Stock Component (60% allocation)')
ax2.set_xlabel('Stock Final Price ($)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bond component
ax3 = axes[1, 0]
ax3.hist(portfolio_results['bond_price'], bins=50, alpha=0.7, 
            color='orange', edgecolor='black')
mean_bond = portfolio_results['bond_price'].mean()
ax3.axvline(mean_bond, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: ${mean_bond:.2f}')
ax3.axvline(117.0, color='gray', linestyle=':', linewidth=2, 
            label='Initial: $117.00')
ax3.set_title('Bond Component (40% allocation)')
ax3.set_xlabel('Bond Final Price ($)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Stock vs Bond correlation
ax4 = axes[1, 1]
ax4.scatter(portfolio_results['stock_price'], portfolio_results['bond_price'],
            alpha=0.3, s=10, c='purple')
ax4.set_xlabel('Stock Final Price ($)')
ax4.set_ylabel('Bond Final Price ($)')
ax4.set_title('Stock vs Bond Correlation')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'portfolio_analysis.png', dpi=300, bbox_inches='tight')
print("    Saved: pic/portfolio_analysis.png")
plt.show()

# ================== PLOT 3: Returns Comparison (Normalized) ==================
    
fig, ax = plt.subplots(figsize=(11, 8))

# Calculate returns
gbm_returns = (gbm_results['final_price'] / S0 - 1) * 100
# jump_returns = (jump_results['final_price'] / S0 - 1) * 100
# regime_returns = (regime_results['final_price'] / S0 - 1) * 100
portfolio_returns = (portfolio_results['portfolio_value'] / PORTFOLIO_INITIAL - 1) * 100

# Plot distributions
ax.hist(gbm_returns, bins=50, alpha=0.3, label='GBM', color='steelblue', density=True)
# ax.hist(jump_returns, bins=50, alpha=0.5, label='Jump Diffusion', color='coral', density=True)
# ax.hist(regime_returns, bins=50, alpha=0.5, label='Regime-Switching', color='mediumseagreen', density=True)
ax.hist(portfolio_returns, bins=50, alpha=0.5, label='Portfolio (60AAPL/40Bond)', color='purple', density=True)

# Add mean lines
ax.axvline(gbm_returns.mean(), color='steelblue', linestyle='--', linewidth=2)
# ax.axvline(jump_returns.mean(), color='coral', linestyle='--', linewidth=2)
# ax.axvline(regime_returns.mean(), color='mediumseagreen', linestyle='--', linewidth=2)
ax.axvline(portfolio_returns.mean(), color='purple', linestyle='--', linewidth=2)

ax.set_xlabel('Return (%)')
ax.set_ylabel('Density')
ax.set_title('Returns Distribution Comparison (Normalized)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir/ 'returns_comparison.png', dpi=300, bbox_inches='tight')
print("    Saved: pic/returns_comparison.png")
plt.show()

# ==================== PLOT 4: Risk-Return Summary ====================
    
fig, ax = plt.subplots(figsize=(10, 7))

models = ['GBM', 'Jump', 'Regime', 'Portfolio']
colors = ['steelblue', 'coral', 'mediumseagreen', 'purple']

# Calculate statistics
stats_data = []
for name, data, initial in [
    ('GBM', gbm_results['final_price'], S0),
    ('Jump', jump_results['final_price'], S0),
    ('Regime', regime_results['final_price'], S0),
    ('Portfolio', portfolio_results['portfolio_value'], PORTFOLIO_INITIAL)
]:
    returns = (data / initial - 1) * 100
    mean_return = returns.mean()
    std_return = returns.std()
    stats_data.append((mean_return, std_return))

# Scatter plot
for i, (model, (ret, risk), color) in enumerate(zip(models, stats_data, colors)):
    ax.scatter(risk, ret, s=300, c=color, alpha=0.7, edgecolors='black', linewidth=2)
    ax.annotate(model, (risk, ret), 
                ha='center', va='center')

ax.set_xlabel('Risk (Std Dev of Returns, %)')
ax.set_ylabel('Expected Return (%)')
ax.set_title('Risk-Return Profile')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir/ 'risk_return_summary.png', dpi=300, bbox_inches='tight')
print("    Saved: pic/risk_return_summary.png")
plt.show()