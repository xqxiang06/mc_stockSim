#ifndef PORTFOLIO_H
#define PORTFOLIO_H
#include "montecarlo_gbm.h"
#include "vasicekBond.h"
#include "correlation.h"
#include <vector>
#include <string>
#include <utility>

/**
 * 3 Fund Portfolio (US Stock + Intl Stock + Bond)
 * 
 * Simulates a portfolio with:
 * - Stock following GBM/Jump/Regime-switching
 * - Bond following Vasicek interest rate model
 * - Fixed allocation (e.g., 42% USstock / 18% Intlstock / 40% bond)
 * - Correlated dynamics via correlation matrix
 */

class Portfolio {
public:
    // constructor
    Portfolio(
        double total_investment,
        double us_stock_S0, // intial stock price
        double us_stock_mu, // stock drift
        double us_stock_sigma, // stock volatility
        double intl_stock_S0,
        double intl_stock_mu,
        double intl_stock_sigma,
        const VasicekParameters& bond_params, //vasicek parameters for bond
        double bond_maturity, // bond maturity in years
        double us_stock_weight, // portfolio weight (0-1)
        double intl_stock_weight,
        double bond_weight,
        const std::vector<std::vector<double>> &corr_matrix, // stocks-bond return correlation
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
    
    // Get component final prices
    double getMeanUSStockFinalPrice() const;
    double getMeanIntlStockFinalPrice() const;
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
    double us_stock_S0, us_stock_mu, us_stock_sigma;
    double intl_stock_S0, intl_stock_mu, intl_stock_sigma;
    VasicekParameters bond_params;
    double bond_maturity;
    
    // Portfolio parameters
    double total_investment;
    double us_stock_weight, intl_stock_weight, bond_weight;
    std::vector<std::vector<double>> corr_matrix;
    double n_us_stock;
    double n_intl_stock;  // Actual number of shares
    double n_bond;   // Actual number of bonds
    
    // Simulation parameters
    double T;
    int n_steps;
    int n_paths;
    double dt;
    
    // Results storage
    std::vector<double> final_portfolio_values;
    std::vector<double> final_us_stock_prices;
    std::vector<double> final_intl_stock_prices;
    std::vector<double> final_bond_prices;
    
    // Helper for percentile calculation
    double percentile(const std::vector<double>& data, double p) const;
};

#endif