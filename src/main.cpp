#include "montecarlo_gbm.h"
#include "csv_reader.h"
#include "european_option.h"
#include "regime_switch.h"
#include "portfolio.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

int main()
{
    try
    { /* User / Model Configuration */
        const std::string csv_file = "data/VOO_STOCK_DATA.csv";

        const int trading_days_per_year = 252;
        int lookback_days = 252;     // last year for estimation
        int n_paths = 1000000;       // Monte Carlo paths
        int n_steps = 252;           // whole year simulation
        double T = 1.0;              // year(s)
        double jump_threshold = 2.5; // Z-score threshold for identifying jumps

        std::stringstream null_stream;
        std::streambuf* old_cout = std::cout.rdbuf();

        /* Read real historical prices */
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

        out_gbm << "path_id,final_price\n";
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

        out_jump << "path_id,final_price\n";
        const auto& final_prices_jump = mc_jump.getFinalPrices();
        for (size_t i = 0; i < std::min(static_cast<std::size_t>(1000), final_prices_jump.size()); ++i)
        {
            out_jump << i << "," << final_prices_jump[i] << "\n";
        }
        out_jump.close();

        
        // Run Regime-Switching Simulation
        std::cout << "\n     REGIME-SWITCHING SIMULATION\n\n";
        
        // Create regime model (calibrates all 6 parameters from real data)
        // calibrateFromData() prints internally: regime counts, μ/σ per regime, P(N→C), P(C→N)
        MarketData regime_data(prices);
        RegimeConfig regime_cfg = RegimeConfig::calibrateFromData(regime_data, 20);
        RegimeSwitching regime_model(regime_cfg);

        // Print regime configuration
        regime_model.printTransitionMatrix();
        
        auto [pi_normal, pi_crash] = regime_model.getStationaryDistribution();
        std::cout << "\nStationary Distribution:\n";
        std::cout << "  Normal: " << std::fixed << std::setprecision(3) << (pi_normal * 100) << "% of time\n";
        std::cout << "  Crash:  " << (pi_crash * 100) << "% of time\n\n";


        // Simulate with regime-switching
        std::vector<double> final_prices_regime;
        final_prices_regime.reserve(n_paths);
        
        std::mt19937 gen_regime(42);
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        double dt = T / n_steps;
        
        for (int path = 0; path < n_paths; path++) {
            double S = S0;
            regime_model.reset();  // Start each path in Normal
            
            for (int step = 0; step < n_steps; step++) {
                auto params = regime_model.getCurrentParameters();
                
                double Z = normal_dist(gen_regime);
                double drift = (params.mu - 0.5 * params.sigma * params.sigma) * dt;
                double diffusion = params.sigma * std::sqrt(dt) * Z;
                
                S *= std::exp(drift + diffusion);
                regime_model.updateRegime();
            }
            
            final_prices_regime.push_back(S);
        }
        
        // Calculate statistics
        double sum_regime = 0.0;
        for (double p : final_prices_regime) sum_regime += p;
        double mean_price_regime = sum_regime / n_paths;
        
        std::vector<double> sorted_regime = final_prices_regime;
        std::sort(sorted_regime.begin(), sorted_regime.end());
        double median_price_regime = sorted_regime[n_paths / 2];
        
        size_t ci_low_idx = static_cast<size_t>(n_paths * 0.025);
        size_t ci_high_idx = static_cast<size_t>(n_paths * 0.975);
        double ci_low_regime = sorted_regime[ci_low_idx];
        double ci_high_regime = sorted_regime[ci_high_idx];
        
        std::cout << "\n===== Regime-Switching Results =====\n";
        std::cout << "Mean final price   : $" << mean_price_regime << "\n";
        std::cout << "Median final price : $" << median_price_regime << "\n";
        std::cout << "95% CI: [$" << ci_low_regime << ", $" << ci_high_regime << "]\n";
        double expected_return_regime = (mean_price_regime / S0 - 1) * 100;
        std::cout << "Expected return: " << expected_return_regime << "%\n";
        
        // Print detailed regime statistics
        regime_model.printStatistics();
        
        // Write results to CSV
        std::ofstream out_regime("data/mc_results_regime.csv");
        if (!out_regime.is_open()) {
            throw std::runtime_error("Cannot write mc_results_regime.csv");
        }
        
        out_regime << "path_id,final_price\n";
        for (size_t i = 0; i < std::min(static_cast<std::size_t>(1000), final_prices_regime.size()); ++i) {
            out_regime << i << "," << final_prices_regime[i] << "\n";
        }
        out_regime.close();

        
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
        out_summary << "Regime," << mean_price_regime << "," << median_price_regime << ","
                    << ci_low_regime << "," << ci_high_regime << "," << expected_return_regime << "\n";
        out_summary.close();

        // ==================== OPTION PRICING ====================
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


        // ========================= PORTFOLIO SIMULATION =========================
        std::cout << "\n\n===== PORTFOLIO SIMULATION (Stock/Bond) =====\n";
        
        // Load US Total Stock Market data
        std::vector<double> us_stock_prices = readLastNPrices("data/VOO_STOCK_DATA.csv", lookback_days);
        auto [us_stock_mu, us_stock_sigma] = ParameterEstimator::estimateFromPrices(us_stock_prices, trading_days_per_year);
        double us_S0 = us_stock_prices.back();
        std::cout << "\nUS Stock (VOO) Parameters:\n";
        std::cout << "  S0 = $" << us_S0 << ", μ = " << us_stock_mu << ", σ = " << us_stock_sigma << "\n";

        // Load Intl Stock Market Data
        std::vector<double> intl_stock_prices = readLastNPrices("data/VXUS_STOCK_DATA.csv", lookback_days);
        auto [intl_stock_mu, intl_stock_sigma] = ParameterEstimator::estimateFromPrices(intl_stock_prices, trading_days_per_year);
        double intl_S0 = intl_stock_prices.back();
        std::cout << "International Stock (VXUS) Parameters:\n";
        std::cout << "  S0 = $" << intl_S0 << ", μ = " << intl_stock_mu << ", σ = " << intl_stock_sigma << "\n";

        // Manual bond parameters
        VasicekParameters bond_params(
            0.15,   // κ (mean reversion speed)
            0.045,  // θ (long-term mean rate 4.5%)
            0.01,   // σ (rate volatility)
            0.045   // r0 (initial rate 4.5%)
        );
        
        double bond_maturity = 10.0;
        double us_stock_weight = 0.60;  // Three-fund allocations
        double intl_stock_weight = 0.20;        
        double bond_weight = 0.20;

        std::vector<std::vector<double>> corr_matrix = {
            {1.0,  0.75, -0.2},  // US Stock:    corr with self, international, bond
            {0.75, 1.0,  -0.1},  // Intl Stock:  corr with US, self, bond
            {-0.2, -0.1,  1.0}   // Bond:        corr with US, international, self
        };

        std::cout << "\nPortfolio Configuration:\n";
        std::cout << "  Allocation: " << (us_stock_weight * 100) << "% US stock / " 
                  << ((intl_stock_weight) * 100) << "% Intl stock /"
                  << ((bond_weight) * 100) << "% Bond\n";
        std::cout << "  Bond: " << bond_maturity << "-year maturity\n";
        std::cout << "  Bond rate: " << (bond_params.r0 * 100) << "%\n";
        
        double TOTAL_INVESTMENT = 10000.0;

        // Create and run portfolio
        Portfolio portfolio(
            TOTAL_INVESTMENT,
            us_S0, us_stock_mu, us_stock_sigma,
            intl_S0, intl_stock_mu, intl_stock_sigma,
            bond_params,
            bond_maturity,
            us_stock_weight, intl_stock_weight, bond_weight,
            corr_matrix,
            T, n_steps, n_paths
        );
        
        portfolio.simulate();
        
        // Collect results
        double mean_portfolio = portfolio.getMeanFinalValue();
        double median_portfolio = portfolio.getMedianFinalValue();
        auto [pf_ci_low, pf_ci_high] = portfolio.getConfidenceInterval(0.95);
        double portfolio_return = (mean_portfolio / TOTAL_INVESTMENT - 1.0) * 100;
        double sharpe = portfolio.getSharpeRatio(r);
        
        std::cout << "\n===== Portfolio Results =====\n";
        std::cout << "Initial value      : $" << std::fixed << std::setprecision(2) << TOTAL_INVESTMENT << "\n";
        std::cout << "Mean final value   : $" << mean_portfolio << "\n";
        std::cout << "Median final value : $" << median_portfolio << "\n";
        std::cout << "95% CI: [$" << pf_ci_low << ", $" << pf_ci_high << "]\n";
        std::cout << "Expected return: " << portfolio_return << "%\n";
        std::cout << "Sharpe Ratio: " << std::setprecision(3) << sharpe << "\n";
        
        // Write results
        portfolio.writeResultsToCSV("data/portfolio_results.csv");
        
        // Add to summary CSV
        std::ofstream out_summary_append("data/mc_summary.csv", std::ios::app);
        if (out_summary_append.is_open()) {
            out_summary_append << "Portfolio," << mean_portfolio << "," << median_portfolio << ","
                              << pf_ci_low << "," << pf_ci_high << "," << portfolio_return << "\n";
            out_summary_append.close();
        }
        
        // Scale stock and bond && Risk reduction rate
        double us_stock_only_shares = TOTAL_INVESTMENT / us_S0;
        double us_stock_ci_width = (ci_high_gbm - ci_low_gbm) * us_stock_only_shares;
        double portfolio_ci_width = pf_ci_high - pf_ci_low;
        double risk_reduction = (1.0 - portfolio_ci_width / us_stock_ci_width) * 100;
    
        std::cout << "Risk reduction     : " << std::setprecision(1) << risk_reduction << "%\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
