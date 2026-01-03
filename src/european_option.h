#ifndef OPTION_H
#define OPTION_H

#include <vector>
#include <string>

class EuropeanOption {
public:
    enum class Type {
        CALL,
        PUT
    };

    // Constructor
    EuropeanOption(double K, double T, double r, Type option_type);

    // Getters
    double getStrike() const { return K; }
    double getMaturity() const { return T; }
    double getRiskFreeRate() const { return r; }
    Type getType() const { return option_type; }

    /* Calculate payoff for given stock price at maturity */
    double payoff(double stock_price) const;

    /**
     * Price option using Jump Diffusion Monte Carlo with existing ParameterEstimatior
     * @param S0 Current stock price
     * @param mu Drift
     * @param sigma Smooth volatility
     * @param lambda Jump intensity
     * @param mu_J Mean jump size
     * @param sigma_J Jump volatility
     * @param n_paths Number of Monte Carlo paths
     * @param n_steps Number of time steps
     * @return Option price 
    */
    double priceJumpDiffusion(double S0, double mu, double sigma,
                              double lambda, double mu_J, double sigma_J,
                              int n_paths, int n_steps) const;

    /**
     * @param final_prices Vector of terminal stock prices from simulation
     * @return Option price
     */
    double priceFromPaths(const std::vector<double>& final_prices) const;

    // Black-Scholes analytical price (for comparison only)
    double blackScholesPrice(double S0, double sigma) const;

private:
    double K;              // Strike price
    double T;              // Time to maturity (years)
    double r;              // Risk-free rate (annualized)
    Type option_type;      // CALL or PUT

    // Helper: Standard normal CDF
    static double normalCDF(double x);
};

#endif
