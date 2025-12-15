#include "montecarlo_gbm.h"
#include <iostream>
#include <fstream>
#include <iomanip>



int main(int argc, char* argv[]) {
    try {
        std::cout << "\n========== Example 1: Using Estimated Parameters ==========" << std::endl;
        
        double S0 = 120.0;      // Initial price
        double mu = 0.3;        // 30% annual return
        double sigma = 0.5;     // 50% annual volatility
        double T = 0.5;         // Half year
        int n_steps = 126;      // Daily steps (trading days)
        int n_paths = 2000000;  // 1 million simulations
        
        MonteCarloGBM mc(S0, mu, sigma, T, n_steps, n_paths);
        mc.simulate();
        
        std::cout << "\n--- Results ---" << std::endl;
        std::cout << "Mean final price: $" << mc.getMeanFinalPrice() << std::endl;
        std::cout << "Median final price: $" << mc.getMedianFinalPrice() << std::endl;
        
        auto [lower, upper] = mc.getConfidenceInterval(0.95);
        std::cout << "95% Confidence Interval: [$" << lower << ", $" << upper << "]" << std::endl;
        
        
        // Example 2: Parameter estimation from historical data
        std::cout << "\n\n========== Example 2: Parameter Estimation ==========" << std::endl;
        
        // Simulated historical prices (in real use, read from CSV)
        std::vector<double> historical_prices = {
            100, 102, 101, 105, 107, 110, 108, 112, 115, 113,
            118, 120, 119, 122, 125, 128, 126, 130, 135, 140
        };
        
        auto [est_mu, est_sigma] = ParameterEstimator::estimateFromPrices(historical_prices, 252);
        
        std::cout << "Estimated parameters from historical data:" << std::endl;
        std::cout << "  mu = " << est_mu << std::endl;
        std::cout << "  sigma = " << est_sigma << std::endl;
        
        // Cap parameters
        double mu_capped = std::min(est_mu, 0.3);
        double sigma_capped = std::min(est_sigma, 0.5);
        
        std::cout << "Capped parameters:" << std::endl;
        std::cout << "  mu = " << mu_capped << std::endl;
        std::cout << "  sigma = " << sigma_capped << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
