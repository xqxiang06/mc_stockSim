#ifndef PORTFOLIO_H
#define PORTFOLIO_H
#include "montecarlo_gbm.h"
#include "vasicekBond.h"
#include "correlation.h"
#include <vector>
#include <string>
#include <utility>

/**
 * Two-Asset Portfolio (Stock + Bond)
 * 
 * Simulates a portfolio with:
 * - Stock following GBM/Jump/Regime-switching
 * - Bond following Vasicek interest rate model
 * - Fixed allocation (e.g., 60% stock / 40% bond)
 * - Correlated dynamics via correlation matrix
 */

class Portfolio {
public:
    // constructor
    Portfolio(
        double stock_S0, // intital stock price
        double stock_mu, // stock drift
        double stock_sigma, // stock volatility
        const VasicekParameters& bond_params, //vasicek parameters for bond
        double bond_maturity, // bond maturity in years
        double correlation, // stock-bond return correlation
        double stock_weight, // portfolio weight in stock (0-1)
        double T, // simulation horizon (years)
        int n_steps, // numbers of time steps
        int n_paths // monte carlo path
    );
    
    // Run portfolio simulation with correlated assets
    void simulate();
    
    // Get portfolio statistics
    double getMeanFinalValue() const;
    double getMedianFinalValue() const;
    std::pair<double, double> getConfidenceInterval(double confidence = 0.95) const;
    
    // Get final prices
    double getMeanStockFinalPrice() const;
    double getMeanBondFinalPrice() const;
    
    // Get final portfolio values across all paths
    const std::vector<double>& getFinalValues() const { return final_portfolio_values; }
    
    // Portfolio-level risk metrics
    double getSharpeRatio(double risk_free_rate) const;
            // It represents the additional amount of return that an investor receives per unit of increase in risk.
    double getMaxDrawdown() const;  // Computed from mean path
    
    // Write results to CSV
    void writeResultsToCSV(const std::string& filename) const;
    
private:
    // Asset parameters
    double stock_S0;
    double stock_mu;
    double stock_sigma;
    VasicekParameters bond_params;
    double bond_maturity;
    
    // Portfolio parameters
    double correlation;
    double stock_weight;
    double bond_weight;
    
    // Simulation parameters
    double T;
    int n_steps;
    int n_paths;
    double dt;
    
    // Results storage
    std::vector<double> final_portfolio_values;
    std::vector<double> final_stock_prices;
    std::vector<double> final_bond_prices;
    
    // Helper for percentile calculation
    double percentile(const std::vector<double>& data, double p) const;
};

//-------------------------------------------------------------------------------------------

// Portfolio calibrator - estimates all parameters from historical data
class PortfolioCalibrator {
public:
    /**
     * Calibrate two-asset portfolio from stock prices and bond yields
     * @param stock_prices Historical stock prices
     * @param bond_yields Historical short-term interest rates (e.g., 3-month T-bill)
     * @param stock_weight Target stock allocation
     * @param trading_days_per_year 252 for daily data
     * @return Configured Portfolio object
     */
    static Portfolio calibrateFromData(
        const std::vector<double>& stock_prices,
        const std::vector<double>& bond_yields,
        double stock_weight,
        double T,
        int n_steps,
        int n_paths,
        int trading_days_per_year = 252
    );
};

#endif