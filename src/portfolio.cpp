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
    double stock_S0,
    double stock_mu,
    double stock_sigma,
    const VasicekParameters& bond_params,
    double bond_maturity,
    double correlation,
    double stock_weight,
    double T,
    int n_steps,
    int n_paths
) : stock_S0(stock_S0), stock_mu(stock_mu), stock_sigma(stock_sigma),
    bond_params(bond_params), bond_maturity(bond_maturity),
    correlation(correlation), stock_weight(stock_weight),
    total_investment(total_investment), 
    T(T), n_steps(n_steps), n_paths(n_paths)
{
    bond_weight = 1.0 - stock_weight;
    dt = T / n_steps;
    
    // Validate inputs
    if (stock_weight < 0.0 || stock_weight > 1.0) {
        throw std::invalid_argument("Stock weight must be in [0, 1]");
    }
    if (correlation < -1.0 || correlation > 1.0) {
        throw std::invalid_argument("Correlation must be in [-1, 1]");
    }
}

void Portfolio::simulate() {
    std::cout << "\n===== Portfolio Simulation Parameters =====\n";
    std::cout << "  Total investment: $" << total_investment << "\n";
    std::cout << "  Allocation: " << (stock_weight*100) << "% stock / "
            << (bond_weight*100) << "% bond\n";
    std::cout << "  Stock: S0=$" << stock_S0 << ", μ=" << stock_mu 
              << ", σ=" << stock_sigma << "\n";
    std::cout << "  Bond: r0=" << bond_params.r0 << ", κ=" << bond_params.kappa
              << ", θ=" << bond_params.theta << ", σ=" << bond_params.sigma << "\n";
    std::cout << "  Correlation: " << correlation << "\n";
    
    // Initialize correlation matrix and Cholesky decomposition
    CorrelationMatrix corr_matrix(correlation);
    
    // Initialize RNG
    std::mt19937 rng(42);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    
    // Pre-allocate result vectors
    final_portfolio_values.resize(n_paths);
    final_stock_prices.resize(n_paths);
    final_bond_prices.resize(n_paths);
    
    // CALCULATE SHARES TO BUY
    double stock_dollars = total_investment * stock_weight;      // e.g. $6000
    double bond_dollars = total_investment * (1 - stock_weight); // e.g. $4000

    // Calculate bond scaling to real bond price ($117)
    VasicekBond temp_bond(bond_params, T, n_steps, bond_maturity);
    double bond_price_L1 = temp_bond.getInitialPrice(); // on a raw scale of 1 (~$0.64)
    double bond_scale = 117.0 / bond_price_L1;          // Scale to $117
    double initial_bond_price = 117.0;

    // Calculate actual shares to buy
    n_stock = stock_dollars / stock_S0;                  // $6000 / $634.15 = 9.46 shares
    n_bond = bond_dollars / initial_bond_price;          // $4000 / $117 = 34.19 bonds

    std::cout << "\n=== Shares Purchased ===\n";
    std::cout << "  Stock: $" << std::fixed << std::setprecision(2) << stock_dollars 
              << " / $" << stock_S0 << " = " << std::setprecision(3) << n_stock << " shares\n";
    std::cout << "  Bond:  $" << std::setprecision(2) << bond_dollars 
              << " / $" << initial_bond_price << " = " << std::setprecision(3) << n_bond << " bonds\n";
    std::cout << "\n  Simulating " << n_paths << " paths over " << n_steps << " steps...\n";
    
    
    // Simulate each path
    double sqrt_dt = std::sqrt(dt);
    double stock_drift = (stock_mu - 0.5 * stock_sigma * stock_sigma) * dt;
    
    for (int path = 0; path < n_paths; ++path) {
        // Generate correlated normals for this path
        std::vector<double> stock_normals(n_steps);
        std::vector<double> bond_normals(n_steps);
        
        for (int step = 0; step < n_steps; ++step) {
            // Generate independent standard normals
            double Z1 = normal_dist(rng);
            double Z2 = normal_dist(rng);
            
            // Apply correlation via Cholesky
            auto [W1, W2] = corr_matrix.generateCorrelated(Z1, Z2);
            
            stock_normals[step] = W1;
            bond_normals[step] = W2;
        }
        
        // Simulate stock (GBM)
        double S = stock_S0;
        for (int step = 0; step < n_steps; ++step) {
            double dW = stock_normals[step] * sqrt_dt;
            S *= std::exp(stock_drift + stock_sigma * dW);
        }
        final_stock_prices[path] = S;
        
        // Simulate bond (Vasicek) and scale
        VasicekBond bond(bond_params, T, n_steps, bond_maturity);
        auto bond_path = bond.simulatePath(&bond_normals);
        double bond_price_T = bond_path.back() * bond_scale;  // Scale to $117 basis
        final_bond_prices[path] = bond_price_T;
        
        // Portfolio value
        final_portfolio_values[path] = n_stock * S + n_bond * bond_price_T;
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

double Portfolio::getMeanStockFinalPrice() const {
    if (final_stock_prices.empty()) return 0.0;
    double sum = std::accumulate(final_stock_prices.begin(), 
                                  final_stock_prices.end(), 0.0);
    return sum / final_stock_prices.size();
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
    VasicekBond temp_bond(bond_params, T, n_steps, bond_maturity);
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
    
    out << "path_id,portfolio_value,stock_price,bond_price\n";
    
    size_t n = std::min(size_t(1000), final_portfolio_values.size());
    for (size_t i = 0; i < n; ++i) {
        out << i << "," 
            << final_portfolio_values[i] << ","
            << final_stock_prices[i] << ","
            << final_bond_prices[i] << "\n";
    }
    
    out.close();
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


// ==================== PortfolioCalibrator Implementation ====================

Portfolio PortfolioCalibrator::calibrateFromData(
    double total_investment,
    const std::vector<double>& stock_prices,
    const std::vector<double>& bond_yields,
    double stock_weight,
    double T,
    int n_steps,
    int n_paths,
    int trading_days_per_year)
{
    if (stock_prices.size() != bond_yields.size()) {
        throw std::invalid_argument("Stock prices and bond yields must have same length");
    }
    
    // Estimate stock parameters
    auto [stock_mu, stock_sigma] = ParameterEstimator::estimateFromPrices(
        stock_prices, trading_days_per_year);
    double stock_S0 = stock_prices.back();
    
    // Estimate bond parameters
    VasicekParameters bond_params = VasicekEstimator::estimateFromRates(
        bond_yields, 1.0 / trading_days_per_year);
    
    // Estimate correlation
    auto stock_returns = ParameterEstimator::computeLogReturns(stock_prices);
    
    // For bonds, compute "returns" as -Δy (yield changes negatively correlate with price)
    std::vector<double> bond_returns;
    bond_returns.reserve(bond_yields.size() - 1);
    for (size_t i = 1; i < bond_yields.size(); ++i) {
        bond_returns.push_back(-(bond_yields[i] - bond_yields[i-1]));
    }
    
    double correlation = CorrelationEstimator::estimateCorrelation(
        stock_returns, bond_returns);
    
    std::cout << "\n===== Calibrated Portfolio Parameters =====\n";
    std::cout << "Stock: μ=" << stock_mu << ", σ=" << stock_sigma 
              << ", S0=$" << stock_S0 << "\n";
    std::cout << "Bond: κ=" << bond_params.kappa << ", θ=" << bond_params.theta
              << ", σ=" << bond_params.sigma << ", r0=" << bond_params.r0 << "\n";
    std::cout << "Correlation (stock-bond): " << correlation << "\n";
    
    return Portfolio(
        total_investment, stock_S0, 
        stock_mu, stock_sigma,
        bond_params, 10.0,  // Default 10-year maturity
        correlation,
        stock_weight,
        T, n_steps, n_paths
    );
}