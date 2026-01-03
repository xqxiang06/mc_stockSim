#include "montecarlo_gbm.h"
#include "csv_reader.h"
#include "european_option.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int main()
{
    try
    { /* User / Model Configuration */
        const std::string csv_file = "data/nvda_stock.csv";

        const int trading_days_per_year = 252;
        int lookback_days = 126;     // last half year for estimation
        int n_paths = 1000000;       // Monte Carlo paths
        int n_steps = 126;           // half year simulation
        double T = 0.5;              // year(s)
        double jump_threshold = 2.5; // Z-score threshold for identifying jumps

        std::stringstream null_stream;
        std::streambuf* old_cout = std::cout.rdbuf();

        /* Read real NVIDIA prices */
        std::vector<double> prices = readLastNPrices(csv_file, lookback_days);

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

        /* Estimate Jump Parameters */
        auto jump_params = ParameterEstimator::estimateJumpParameters(
            prices, jump_threshold, trading_days_per_year);

        std::cout << "\n===== Jump Diffusion Parameters =====\n";
        std::cout << "Jump intensity (lambda) = " << jump_params.lambda << " jumps/year\n";
        std::cout << "Mean jump size (mu_J) = " << jump_params.mu_J << " (" << (jump_params.mu_J*100) << "%)\n";
        std::cout << "Jump volatility (sigma_J) = " << jump_params.sigma_J << " (" << (jump_params.sigma_J*100) << "%)\n";
        std::cout << "Smooth volatility (sigma_smooth) = " << jump_params.sigma_smooth << " (" << (jump_params.sigma_smooth*100) << "%)\n";
        
        /* Run Pure GBM simulation */
        std::cout << "\n        PURE GBM SIMULATION\n";
        MonteCarloGBM mc_gbm(S0, mu, sigma, T, n_steps, n_paths); // without jump parameters
        mc_gbm.simulate();

        /* Collect results */
        double mean_price_gbm = mc_gbm.getMeanFinalPrice();
        double median_price_gbm = mc_gbm.getMedianFinalPrice();
        auto [ci_low_gbm, ci_high_gbm] = mc_gbm.getConfidenceInterval(0.95);

        std::cout << "\n===== Pure GBM Results =====\n";
        std::cout << "Mean final price   : " << mean_price_gbm << "\n";
        std::cout << "Median final price : " << median_price_gbm << "\n";
        std::cout << "95% CI: [" << ci_low_gbm << ", " << ci_high_gbm << "]\n";
        // Expected return
        double expected_return_gbm = (mean_price_gbm / S0 - 1) * 100;
        std::cout << "Expected return: " << expected_return_gbm << "%\n";

        /* Run Jump Diffusion Simulation */
        std::cout << "\n      JUMP DIFFUSION SIMULATION\n";
        
        MonteCarloGBM mc_jump(S0, mu, jump_params.sigma_smooth, T, n_steps, n_paths,
                              jump_params.lambda, jump_params.mu_J, jump_params.sigma_J);
        mc_jump.simulate();

        /* Collect results */
        double mean_price_jump = mc_jump.getMeanFinalPrice();
        double median_price_jump = mc_jump.getMedianFinalPrice();
        auto [ci_low_jump, ci_high_jump] = mc_jump.getConfidenceInterval(0.95);

        std::cout << "\n===== Jump Diffusion Results =====\n";
        std::cout << "Mean final price   : $" << mean_price_jump << "\n";
        std::cout << "Median final price : $" << median_price_jump << "\n";
        std::cout << "95% CI: [$" << ci_low_jump << ", $" << ci_high_jump << "]\n";
        double expected_return_jump = (mean_price_jump / S0 - 1) * 100;
        std::cout << "Expected return: " << expected_return_jump << "%\n";

        /* Write results to CSV */
        std::ofstream out_gbm("data/mc_results_gbm.csv");
        if (!out_gbm.is_open()) {
            throw std::runtime_error("Cannot write mc_results_gbm.csv");
        }

        out_gbm << "path_id, final_price\n";
        const auto& final_prices_gbm = mc_gbm.getFinalPrices(); // get final price
        for (size_t i = 0; i < std::min(static_cast<std::size_t>(1000), final_prices_gbm.size()); ++i)
        {
            out_gbm << i << "," << final_prices_gbm[i] << "\n";
        }
        out_gbm.close();

        // Jump Diffusion results
        std::ofstream out_jump("data/mc_results_jump.csv");
        if (!out_jump.is_open()) {
            throw std::runtime_error("Cannot write mc_results_jump.csv");
        }

        out_jump << "path_id, final_price\n";
        const auto& final_prices_jump = mc_jump.getFinalPrices();
        for (size_t i = 0; i < std::min(static_cast<std::size_t>(1000), final_prices_jump.size()); ++i)
        {
            out_jump << i << "," << final_prices_jump[i] << "\n";
        }
        out_jump.close();

        // Summary Statstics
        std::ofstream out_summary("data/mc_summary.csv");
        if (!out_summary.is_open()) {
            throw std::runtime_error("Cannot write mc_summary.csv");
        }

        out_summary << "model,mean_price,median_price,ci_low,ci_high,expected_return\n";
        out_summary << "GBM," << mean_price_gbm << "," << median_price_gbm << ","
                    << ci_low_gbm << "," << ci_high_gbm << "," << expected_return_gbm << "\n";
        out_summary << "Jump," << mean_price_jump << "," << median_price_jump << ","
                    << ci_low_jump << "," << ci_high_jump << "," << expected_return_jump << "\n";
        out_summary.close();

        // ==================== ADD OPTION PRICING ====================
        std::cout << "\n\n===== OPTION PRICING =====\n";
        
        // Option parameters (can be changed)
        double r = 0.045;  // 4.5% risk-free rate (T-bill)
        std::cout << "[Risk-Neutral Measure: risk-free rate r=" << (r*100) << "%]\n";
        
        // Price a few common options
        struct OptionContract {
            double K;
            double T;
            std::string name;
            EuropeanOption::Type type;
        };
        
        std::vector<OptionContract> options = {
        // call constructor
            {S0 * 1.05, 0.25, "3-month 5% OTM Call", EuropeanOption::Type::CALL},
            {S0, 0.5, "6-month ATM Call", EuropeanOption::Type::CALL},
            {S0 * 0.95, 0.25, "3-month 5% OTM Put", EuropeanOption::Type::PUT},
            {S0, 0.5, "6-month ATM Put", EuropeanOption::Type::PUT}
        };
        
        std::ofstream out_options("data/option_prices.csv");
        if (!out_options.is_open()) {
            throw std::runtime_error("Cannot write option_prices.csv");
        }
        
        out_options << "contract,type,strike,maturity,jd_price,bs_price,jump_premium_percent\n";
        
        for (const auto& opt : options) {
            // Suppress MC output during option pricing
            std::cout.rdbuf(null_stream.rdbuf());

            EuropeanOption option(opt.K, opt.T, r, opt.type);
            
            int opt_steps = static_cast<int>(opt.T * trading_days_per_year);
            
            double jd_price = option.priceJumpDiffusion(
                S0, mu, jump_params.sigma_smooth,
                jump_params.lambda, jump_params.mu_J, jump_params.sigma_J,
                100000, opt_steps
            );
            // Black-Scholes price (for comparison)
            double bs_price = option.blackScholesPrice(S0, jump_params.sigma_smooth);
            double premium_pct = ( (jd_price - bs_price) / bs_price ) * 100;
                    /* the extra return investors demand for taking on the risk of sudden, large, unpredictable price changes (jumps) 
                        in stocks or the overall market, beyond normal volatility (compensation for "tail risk") */

            std::cout.rdbuf(old_cout);  // Restore cout for printing
            std::cout << opt.name << ": $" << std::setprecision(2) << jd_price << "\n";
            
            out_options << opt.name << ","
                        << (opt.type == EuropeanOption::Type::CALL ? "CALL" : "PUT") << ","
                        << opt.K << "," << opt.T << "," << jd_price << "," << bs_price << ","
                        << premium_pct << "\n";
        }
        out_options.close();
        std::cout << "\nResults written to csv files\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
