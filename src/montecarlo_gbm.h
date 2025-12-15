#ifndef MONTECARLO_GBM_H
#define MONTECARLO_GBM_H

#include <vector>
#include <random>
#include <cmath>

/*
Monte Carlo Simulation for Stock Prices using Geometric Brownian Motion
    Model: S(t) = S₀ × exp((μ - σ²/2)t + σW(t))
    where W(t) is a Wiener process (Brownian motion) */

class MonteCarloGBM {
    public:
    // Constructor
    MonteCarloGBM(double S0, double mu, double sigma, double T, int n_steps, int n_paths);
    
    // Run Monte Carlo simulation
    void simulate();

    // Get results
    const std::vector<std::vector<double>>& getPaths() const { return paths; }
    const std::vector<double>& getTimeGrid() const { return time_grid; }
    std::vector<double> getFinalPrices() const;
    
    // Statistical analysis
    double getMeanFinalPrice() const;
    double getMedianFinalPrice() const;
    std::pair<double, double> getConfidenceInterval(double confidence = 0.95) const;
    
    // Get parameters
    double getS0() const { return S0; }
    double getMu() const { return mu; }
    double getSigma() const { return sigma; }
    double getT() const { return T; }
    int getNSteps() const { return n_steps; }
    int getNPaths() const { return n_paths; }

private:
    // Model parameters
    double S0;        // Initial stock price
    double mu;        // Expected annual return (drift)
    double sigma;     // Annual volatility
    double T;         // Time horizon (in years)
    int n_steps;      // Number of time steps
    int n_paths;      // Number of simulation paths
    
    // Simulation results
    std::vector<std::vector<double>> paths;  // paths[step][time]
    std::vector<double> time_grid;           // time points
    
    // Random number generation
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
    
    // Helper functions
    void generateTimeGrid();
    double percentile(const std::vector<double>& data, double p) const;
};

 
// Utility class for estimating GBM parameters from historical data
class ParameterEstimator {
public:
    // Estimate mu and sigma from price series
    static std::pair<double, double> estimateFromPrices(
        const std::vector<double>& prices, 
        int trading_days_per_year = 252
    );
    
private:
    static std::vector<double> computeLogReturns(const std::vector<double>& prices);
    static double mean(const std::vector<double>& data);
    static double stddev(const std::vector<double>& data);
};

#endif
