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


# ==================== 1. Price Distribution Comparison ====================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Simulation of Final Price Distributions', fontsize=16, fontweight='bold')

# GBM
ax1 = axes[0, 0]
ax1.hist(gbm_results['final_price'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(summary[summary['model'] == 'GBM']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax1.set_title('Pure Geometric Brownian Motion', fontweight='bold')
ax1.set_xlabel('Final Price ($)')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Jump Diffusion
ax2 = axes[0, 1]
ax2.hist(jump_results['final_price'], bins=50, alpha=0.7, color='coral', edgecolor='black')
ax2.axvline(summary[summary['model'] == 'Jump']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax2.set_title('Jump Diffusion (Merton Model)', fontweight='bold')
ax2.set_xlabel('Final Price ($)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Regime-Switching
ax3 = axes[1, 0]
ax3.hist(regime_results['final_price'], bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
ax3.axvline(summary[summary['model'] == 'Regime']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax3.set_title('Regime-Switching (Normal-Crash)', fontweight='bold')
ax3.set_xlabel('Final Price ($)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Portfolio (Stock + Bond)
ax4 = axes[1, 1]
ax4.hist(portfolio_results['portfolio_value'], bins=50, alpha=0.7, color='purple', edgecolor='black')
ax4.axvline(summary[summary['model'] == 'Portfolio']['median_price'].values[0], 
            color='red', linestyle='--', linewidth=2, label='Median')
ax4.set_title('60/40 Portfolio (Stock + Bond)', fontweight='bold')
ax4.set_xlabel('Final Portfolio Value ($)')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'price_distributions.png', dpi=300, bbox_inches='tight')
print(f"    Saved: pic/price_distributions.png")