#include "european_option.h"
#include "montecarlo_gbm.h" // using exist mc_gbm to simulate for risk-neutral price
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>


EuropeanOption::EuropeanOption(double K, double T, double r, Type option_type)
    : K(K), T(T), r(r), option_type(option_type)
{
    if (K <= 0 || T <= 0) throw std::invalid_argument
                            ("Strike price and Time to maturity must be positive");
}

double EuropeanOption::payoff(double stock_price) const
{                     // option value
    if (option_type == Type::CALL) {
        // If the price is higher than the exercise price ~ profit
        //                 lower than the exercise price ~ abandoned ~ profit=0
        return std::max(stock_price - K, 0.0);
    } else {
        return std::max(K - stock_price, 0.0); // put option
    }
}

double EuropeanOption::normalCDF(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}                   // complementary error function of a given value

// Comparison: Black-Scholes model
double EuropeanOption::blackScholesPrice(double S0, double sigma) const
{
    if (sigma <= 0 || S0 <= 0) {
        throw std::invalid_argument("S0 and sigma must be positive");
    }

    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) 
                / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    if (option_type == Type::CALL) {
        return S0 * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
    } else {
        return K * std::exp(-r * T) * normalCDF(-d2) - S0 * normalCDF(-d1);
    }
}

double EuropeanOption::priceFromPaths(const std::vector<double>& final_prices) const
{
    if (final_prices.empty()) {
        throw std::invalid_argument("final_prices vector is empty");
    }

    // Calculate average discounted payoff
    double sum_payoffs = 0.0;
    for (double S_T : final_prices) {
        sum_payoffs += payoff(S_T);
    }

    double avg_payoff = sum_payoffs / final_prices.size();
    // option value at time t=0: e^(-rT) * mean.(payoff)
                            // deduce from back to start
    return std::exp(-r * T) * avg_payoff;
}

double EuropeanOption::priceJumpDiffusion(double S0, double mu, double sigma,
                                         double lambda, double mu_J, double sigma_J,
                                         int n_paths, int n_steps) const
{
    // use MonteCarloGBM class to get final_prices
    MonteCarloGBM mc(S0, mu, sigma, T, n_steps, n_paths, 
                     lambda, mu_J, sigma_J,
                     true, r); // introduce risk_neutral=true & r
    
    // Run simulation
    mc.simulate();
    
    // Get final prices and price the option
    const auto& final_prices = mc.getFinalPrices();
    return priceFromPaths(final_prices);
}
