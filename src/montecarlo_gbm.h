#ifndef MONTECARLO_GBM_H
#define MONTECARLO_GBM_H

#include <vector>
#include <cmath>
#include <utility>

/*
Monte Carlo Simulation for Stock Prices using Geometric Brownian Motion
    Model: S(t) = S₀ × exp((μ - σ²/2)t + σW(t))
ADD: Jump-diffusion (Poisson Process) to accept sudden price changes
    Model: dS = μS dt + σS dW + S(J-1) dN */

class MonteCarloGBM {
    public:
    // ADD: Constructor with jump parameters
    MonteCarloGBM(double S0, double mu, double sigma, double T, int n_steps, int n_paths,
                  double lambda = 0.0, double mu_J = 0.0, double sigma_J = 0.0,
                  bool risk_neutral = false, double r = 0.0) ;
                    // use for option
    
    // Run Monte Carlo simulation
    void simulate();

    // Get results
    const std::vector<std::vector<double>>& getPaths() const { return paths; }
    const std::vector<double>& getTimeGrid() const { return time_grid; }
    const std::vector<double>& getFinalPrices() const;
    
    // Statistical analysis
    double getMeanFinalPrice() const;
    double getMedianFinalPrice() const;
    std::pair<double, double> getConfidenceInterval(double confidence = 0.95) const;
    static double percentile(const std::vector<double>& data, double p);
    
    // Get parameters
    double getS0() const { return S0; }
    double getMu() const { return mu; }
    double getSigma() const { return sigma; }
    double getT() const { return T; }
    int getNSteps() const { return n_steps; }
    int getNPaths() const { return n_paths; }
    // ADD: Get jump parameters
    double getLambda() const { return lambda; }
    double getMuJ() const { return mu_J; }
    double getSigmaJ() const { return sigma_J; }
    bool hasJumps() const { return lambda > 0.0; }

private:
    // Model parameters
    double S0;        // Initial stock price
    double mu;        // Expected annual return (drift)
    double sigma;     // Annual volatility
    double T;         // Time horizon (in years)
    int n_steps;      // Number of time steps
    int n_paths;      // Number of simulation paths
    // ADD: Jump parameters
    double lambda;    // Jump intensity (jumps/year)
    double mu_J;      // Mean of log-jump size
    double sigma_J;   // Std dev of log-jump size
    // use for option
    bool risk_neutral;
    double r;
    
    // Simulation results
    std::vector<std::vector<double>> paths;  // paths[path][step]
    std::vector<double> time_grid;           // time points
    std::vector<double> final_prices;
    
    // Helper functions
    void generateTimeGrid();
};

 
/* Utility class for estimating GBM parameters from historical data */
class ParameterEstimator {
public:
    // Estimate mu and sigma from price series
    static std::pair<double, double> estimateFromPrices(
        const std::vector<double>& prices, 
        int trading_days_per_year = 252
    );

    struct JumpParameters {
        double lambda;         // Jump intensity (jumps per year)
        double mu_J;           // Mean log-jump size
        double sigma_J;        // Jump size volatility
        double sigma_smooth;   // Volatility without jumps (use this for GBM sigma)
    };

    static JumpParameters estimateJumpParameters(
        const std::vector<double>& prices,
        double threshold = 2.5,
        int trading_days_per_year = 252
    );

    static std::vector<double> computeLogReturns(const std::vector<double>& prices);
    
private:
    static double mean(const std::vector<double>& data);
    static double stddev(const std::vector<double>& data);
};

#endif