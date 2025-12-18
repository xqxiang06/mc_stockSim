#include "montecarlo_gbm.h"
#include "csv_reader.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char* argv[])
{
    try
    { /* User / Model Configuration */

        const std::string csv_file = "nvda_stock.csv";

        const int trading_days_per_year = 252;
        int lookback_days = 126;    // last half year for estimation
        int n_paths = 1000000;      // Monte Carlo paths
        int n_steps = 126;          // half year simulation
        double T = 0.5;             // year(s)

        /* Read real NVIDIA prices */

        std::vector<double> prices =
            readLastNPrices(csv_file, lookback_days);

        double S0 = prices.back(); // last observed price

        std::cout << "Loaded " << prices.size() << " historical prices\n";
        std::cout << "Initial price S0 = " << S0 << "\n";

        /* Estimate μ and σ from data */

        auto [mu_est, sigma_est] =
            ParameterEstimator::estimateFromPrices(
                prices, trading_days_per_year);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Estimated mu = " << mu_est << " (" << (mu_est*100) << "% annual return)\n";
        std::cout << "Estimated sigma = " << sigma_est << " (" << (sigma_est*100) << "% annual volatility)\n";

        // Cap to reasonable values
        double mu = std::min(mu_est, 0.3);
        double sigma = std::min(sigma_est, 0.5);

        /* Run Monte Carlo simulation */

        MonteCarloGBM mc(S0, mu, sigma, T, n_steps, n_paths);

        mc.simulate();

        /* Collect results */

        const auto& final_prices = mc.getFinalPrices();
        double mean_price = mc.getMeanFinalPrice();
        double median_price = mc.getMedianFinalPrice();
        auto [ci_low, ci_high] = mc.getConfidenceInterval(0.95);

        std::cout << "\n===== Monte Carlo Results =====\n";
        std::cout << "Mean final price   : " << mean_price << "\n";
        std::cout << "Median final price : " << median_price << "\n";
        std::cout << "95% CI: [" << ci_low << ", " << ci_high << "]\n";
        // Expected return
        double expected_return = (mean_price / S0 - 1) * 100;
        std::cout << "Expected return: " << expected_return << "%\n";

        /* Write results to CSV */

        std::ofstream out("mc_results.csv");
        if (!out.is_open())
        {
            throw std::runtime_error("Cannot write mc_results.csv");
        }

        out << "path_id, final_price\n";
        for (size_t i = 0; i < std::min(1000, (int)final_prices.size()); ++i)
        {
            out << i << "," << final_prices[i] << "\n";
        }

        out.close();

        std::cout << "\nResults written to mc_results.csv\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}