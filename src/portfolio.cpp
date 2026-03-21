#include "portfolio.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <iostream>

Portfolio::Portfolio(
    double total_investment,
    double us_stock_S0, double us_stock_mu, double us_stock_sigma,
    double intl_stock_S0, double intl_stock_mu, double intl_stock_sigma,
    const VasicekParameters& bond_params,
    double bond_maturity,
    double us_stock_weight, double intl_stock_weight, double bond_weight,
    const std::vector<std::vector<double>>& corr_matrix,
    double T,
    int n_steps,
    int n_paths
) : us_stock_S0(us_stock_S0), us_stock_mu(us_stock_mu), us_stock_sigma(us_stock_sigma),
    intl_stock_S0(intl_stock_S0), intl_stock_mu(intl_stock_mu), intl_stock_sigma(intl_stock_sigma),
    bond_params(bond_params), bond_maturity(bond_maturity),
    total_investment(total_investment),
    us_stock_weight(us_stock_weight),
    intl_stock_weight(intl_stock_weight),
    bond_weight(bond_weight),
    corr_matrix(corr_matrix),
    T(T), n_steps(n_steps), n_paths(n_paths)
{
    dt = T / n_steps;
    
    // Validate inputs
    double weight_sum = us_stock_weight + intl_stock_weight + bond_weight;
    if (std::abs(weight_sum - 1.0) > 1e-6) {
        throw std::invalid_argument("Portfolio weights must sum to 1.0");
    }
    if (us_stock_weight < 0.0 || intl_stock_weight < 0.0 || bond_weight < 0.0) {
        throw std::invalid_argument("Weights must be non-negative");
    }
}

void Portfolio::simulate() {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n===== Portfolio Simulation Parameters =====\n";
    std::cout << "  Total investment: $" << total_investment << "\n";
    std::cout << "  Allocation: \n"; 
    std::cout << "    US Stocks:      " << (us_stock_weight * 100) << "%\n"
              << "    Intl Stocks:    " << (intl_stock_weight * 100) << "%\n"
              << "    Bonds:          " << (bond_weight*100) << "%\n";
    std::cout << "  US Stock:   S0=$" << us_stock_S0 << ", μ=" << us_stock_mu 
              << ", σ=" << us_stock_sigma << "\n";
    std::cout << "  Intl Stock: S0=$" << intl_stock_S0 << ", μ=" << intl_stock_mu 
              << ", σ=" << intl_stock_sigma << "\n";
    std::cout << "  Bond: r0=" << bond_params.r0 << ", κ=" << bond_params.kappa
              << ", θ=" << bond_params.theta << ", σ=" << bond_params.sigma << "\n";
    
    // Initialize correlation matrix and Cholesky decomposition
    CorrelationMatrix corr3d(corr_matrix);
    
    // Initialize RNG
    std::mt19937 rng(42);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // Pre-allocate result vectors
    final_portfolio_values.resize(n_paths);
    final_us_stock_prices.resize(n_paths);
    final_intl_stock_prices.resize(n_paths);
    final_bond_prices.resize(n_paths);
    
    // Calculate dollar allocations
    double us_stock_dollars = total_investment * us_stock_weight;
    double intl_stock_dollars = total_investment * intl_stock_weight;
    double bond_dollars = total_investment * bond_weight;

    // // Bond price: use model's raw zero-coupon price directly (face value = $1)
    VasicekBond temp_bond(bond_params, T, n_steps, bond_maturity);
    double initial_bond_price = temp_bond.getInitialPrice(); // [0,1]
    double bond_scale = 1.0;

    // Calculate shares to buy
    n_us_stock = us_stock_dollars / us_stock_S0;         // $4200 / $634.15 = 6.62 shares
    n_intl_stock = intl_stock_dollars / intl_stock_S0;   // $1800 / $78 = 23 shares
    n_bond = bond_dollars / initial_bond_price;          // $4000 / units of $1 face-value bonds

    // Display only — convert to ETF-like price for printing
    double display_scale = 74.0 / initial_bond_price;

    std::cout << "\n=== Shares Purchased ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  US Stock: $" << us_stock_dollars << " / $" << us_stock_S0 
              << " = " << n_us_stock << " shares\n";
    std::cout << "  Intl Stock: $" << intl_stock_dollars << " / $" << intl_stock_S0 
              << " = " << n_intl_stock << " shares\n";
    std::cout << "  Bond:  $" << bond_dollars << " / $" << initial_bond_price 
              << " = " << n_bond << " bonds\n";
    std::cout << "  Bond initial price: $" << initial_bond_price * display_scale 
              << " (ETF-scaled)\n";
    std::cout << "\n  Simulating " << n_paths << " paths over " << n_steps << " steps...\n";
    
    
    // Simulate each path
    double sqrt_dt = std::sqrt(dt);
    double us_stock_drift = (us_stock_mu - 0.5 * us_stock_sigma * us_stock_sigma) * dt;
    double intl_stock_drift = (intl_stock_mu - 0.5 * intl_stock_sigma * intl_stock_sigma) * dt;

    // sample path to record intermediate values (for python plotting)
    const int sample_every = 1000;
    const int n_sampled = n_paths / sample_every;
    sampled_paths.assign(n_sampled, std::vector<double>(n_steps + 1)); // [pathId][step]

    for (int path = 0; path < n_paths; ++path) {
        // Generate correlated normals for this path
        std::vector<double> us_stock_normals(n_steps);
        std::vector<double> intl_stock_normals(n_steps);
        std::vector<double> bond_normals(n_steps);
        
        for (int step = 0; step < n_steps; ++step) {
            // Generate independent standard normals
            double Z1 = normal_dist(rng);
            double Z2 = normal_dist(rng);
            double Z3 = normal_dist(rng);
            
            // Apply correlation via Cholesky
            auto [W1, W2, W3] = corr3d.generateCorrelated(Z1, Z2, Z3);
            
            us_stock_normals[step] = W1;
            intl_stock_normals[step] = W2;
            bond_normals[step] = W3;
        }
        
        // Simulate bond (Vasicek) upfront (gives all steps at once)
        VasicekBond bond(bond_params, T, n_steps, bond_maturity);
        auto bond_path = bond.simulatePath(&bond_normals); // size = n_steps + 1

        bool is_sampled = (path % sample_every == 0);
        int  sample_idx = path / sample_every;

        // Step 0: initial values
        if (is_sampled)
            sampled_paths[sample_idx][0] = n_us_stock * us_stock_S0
                                         + n_intl_stock * intl_stock_S0
                                         + n_bond * bond_path[0] * bond_scale;

        // Simulate stocks step by step
        // (merged into one loop for intermediate recording)
        double S_us = us_stock_S0;
        double S_intl = intl_stock_S0;
        
        for (int step = 0; step < n_steps; ++step) {
            // double dW = us_stock_normals[step] * sqrt_dt;
            S_us *= std::exp(us_stock_drift + us_stock_sigma
                                            * us_stock_normals[step] * sqrt_dt);
            S_intl *= std::exp(intl_stock_drift + intl_stock_sigma
                                                * intl_stock_normals[step] * sqrt_dt);
            
            if (is_sampled) {
                double bond_price = bond_path[step + 1] * bond_scale;
                sampled_paths[sample_idx][step + 1] =
                    n_us_stock * S_us + n_intl_stock * S_intl + n_bond * bond_price;
            }
        }
        
        // Portfolio value
        final_us_stock_prices[path]   = S_us;
        final_intl_stock_prices[path] = S_intl;
        final_bond_prices[path]       = bond_path.back() * bond_scale;
        final_portfolio_values[path]  = n_us_stock * S_us + n_intl_stock * S_intl
                                        + n_bond * final_bond_prices[path];
    }
}

double Portfolio::getMeanFinalValue() const {
    if (final_portfolio_values.empty()) return 0.0;
    double sum = std::accumulate(final_portfolio_values.begin(), 
                                  final_portfolio_values.end(), 0.0);
    return sum / final_portfolio_values.size();
}

double Portfolio::getMedianFinalValue() const {
    return percentile(final_portfolio_values, 0.5);
}

std::pair<double, double> Portfolio::getConfidenceInterval(double confidence) const {
    double lower_p = (1 - confidence) * 0.5;
    double upper_p = 1 - lower_p;
    
    double lower = percentile(final_portfolio_values, lower_p);
    double upper = percentile(final_portfolio_values, upper_p);
    
    return {lower, upper};
}

double Portfolio::getMeanUSStockFinalPrice() const {
    if (final_us_stock_prices.empty()) return 0.0;
    double sum = std::accumulate(final_us_stock_prices.begin(), 
                                  final_us_stock_prices.end(), 0.0);
    return sum / final_us_stock_prices.size();
}

double Portfolio::getMeanIntlStockFinalPrice() const {
    if (final_intl_stock_prices.empty()) return 0.0;
    double sum = std::accumulate(final_intl_stock_prices.begin(), 
                                  final_intl_stock_prices.end(), 0.0);
    return sum / final_intl_stock_prices.size();
}

double Portfolio::getMeanBondFinalPrice() const {
    if (final_bond_prices.empty()) return 0.0;
    double sum = std::accumulate(final_bond_prices.begin(), 
                                  final_bond_prices.end(), 0.0);
    return sum / final_bond_prices.size();
}

double Portfolio::getSharpeRatio(double risk_free_rate) const {
    if (final_portfolio_values.empty()) return 0.0;
    
    // Calculate annualized return
    double mean_final = getMeanFinalValue();
    double total_return = (mean_final / total_investment) - 1.0;
    double annualized_return = std::pow(1.0 + total_return, 1.0 / T) - 1.0;
    
    // Calculate annualized volatility
    std::vector<double> returns;
    returns.reserve(final_portfolio_values.size());
    for (double final_val : final_portfolio_values) {
        double ret = (final_val / total_investment) - 1.0;
        double ann_ret = std::pow(1.0 + ret, 1.0 / T) - 1.0;
        returns.push_back(ann_ret);
    }
    
    double mean_ret = annualized_return;
    double var_sum = 0.0;
    for (double ret : returns) {
        var_sum += (ret - mean_ret) * (ret - mean_ret);
    }
    if (returns.size() < 2) return 0.0;
    double volatility = std::sqrt(var_sum / (returns.size() - 1));
    
    if (volatility < 1e-10) return 0.0;
    
    return (annualized_return - risk_free_rate) / volatility; // the Sharpe Ratio formula
}

double Portfolio::getMaxDrawdown() const {
    // For simplicity, compute from average path
    // (proper implementation would track each path's max drawdown)
    return 0.0;  // TODO: implement if needed
}

void Portfolio::writeResultsToCSV(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot write to " + filename);
    }
    
    out << "path_id,portfolio_value,us_stock_price, intl_stock_price,bond_price\n";
    
    size_t n = std::min(size_t(1000), final_portfolio_values.size());
    for (size_t i = 0; i < n; ++i) {
        out << i << "," 
            << final_portfolio_values[i] << ","
            << final_us_stock_prices[i] << ","
            << final_intl_stock_prices[i] << ","
            << final_bond_prices[i] << "\n";
    }
    
    out.close();

    // Derive paths to a new file "data/portfolio_paths.csv"
    std::string paths_file = filename;
    size_t pos = paths_file.rfind(".csv");
    if (pos != std::string::npos) paths_file.replace(pos, 4, "_paths.csv");
    else paths_file += "_paths.csv";

    std::ofstream pout(paths_file);
    if (!pout.is_open())
        throw std::runtime_error("Cannot write to " + paths_file);

    pout << "path_id,step,portfolio_value\n";
    for (size_t p = 0; p < sampled_paths.size(); ++p) {
        for (size_t s = 0; s < sampled_paths[p].size(); ++s) {
            pout << (p * 1000) << "," << s << "," << sampled_paths[p][s] << "\n";
        }
    }
    pout.close();
}

double Portfolio::percentile(const std::vector<double>& data, double p) const {
    if (data.empty()) return 0.0;
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    double index = p * (sorted_data.size() - 1);
    int lower_idx = static_cast<int>(std::floor(index));
    int upper_idx = static_cast<int>(std::ceil(index));
    
    if (lower_idx == upper_idx) {
        return sorted_data[lower_idx];
    }
    
    double weight = index - lower_idx;
    return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight;
}