#include "regime_switch.h"
#include "montecarlo_gbm.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>

// Helper functions for calibration (in anonymous namespace - private)
namespace {
    // Compute rolling volatility: vol_t = std( 1/(window_size) * sum(r_t - r_mean)^2 ) × √252
    //                                                using log_returns
    std::vector<double> computeRollingVolatility(
        const std::vector<double>& returns, 
        int window // the last N days
    ) {
        std::vector<double> volatilities;
        volatilities.reserve(returns.size());
        
        for (size_t i = 0; i < returns.size(); ++i) {
            size_t start = (i >= static_cast<size_t>(window)) ? i - window + 1 : 0; // first index(day) included in the window
            size_t count = i - start + 1; // numbers of data points in the current window
            
            // Calculate mean
            double sum = 0.0;
            for (size_t j = start; j <= i; ++j) {
                sum += returns[j];
            }
            double mean = sum / count;
            
            // Calculate variance
            double var_sum = 0.0;
            for (size_t j = start; j <= i; ++j) {
                double diff = returns[j] - mean;
                var_sum += diff * diff;
            }
            double variance = var_sum / count;
            double vol = std::sqrt(variance) * std::sqrt(252.0);  // Annualized
            
            volatilities.push_back(vol);
        }
        
        return volatilities;
    }

    // K-means clustering (k=2) to separate low/high volatility regimes
    std::vector<int> kMeansClustering(const std::vector<double>& volatilities) {
        if (volatilities.empty()) {
            return {};
        }
        
        // Initialize centroids: min and max volatilities
        auto minmax = std::minmax_element(volatilities.begin(), volatilities.end());
        double c0 = *minmax.first;   // Low volatility centroid (Normal)
        double c1 = *minmax.second;  // High (Crush)
        
        std::vector<int> assignments(volatilities.size(), 0); // assume every point starts at 0
        bool changed = true;
        int max_iterations = 100;
        int iter = 0;
        
        while (changed && iter < max_iterations) { // main loop: until no more changes or exceeds the upper limit
            changed = false;
            ++iter;
            
            // Assignment step: assign each point to nearest centroid
            for (size_t i = 0; i < volatilities.size(); ++i) {
                double dist0 = std::abs(volatilities[i] - c0);
                double dist1 = std::abs(volatilities[i] - c1);
                int new_assignment = (dist0 < dist1) ? 0 : 1; // zi ​= arg( min​∣ vi​−ck​ ∣ )
                
                if (new_assignment != assignments[i]) {
                    assignments[i] = new_assignment;
                    changed = true;
                }
            }
            
            // Update centroids: mean of assigned points
            double sum0 = 0.0, sum1 = 0.0;
            int count0 = 0, count1 = 0;
            
            for (size_t i = 0; i < volatilities.size(); ++i) {
                if (assignments[i] == 0) {
                    sum0 += volatilities[i]; // cluster for 0-normal
                    count0++;
                } else {
                    sum1 += volatilities[i]; // cluster for 1-crush
                    count1++;
                }
            }
            
            if (count0 > 0) c0 = sum0 / count0;
            if (count1 > 0) c1 = sum1 / count1;
        }
        
        return assignments;
    }

    // Estimate parameters for a specific regime
    void estimateRegimeParams(
        const std::vector<double>& returns,
        const std::vector<int>& regime_sequence, // the regime tag corresponding to each day
        int regime_id, // which regime (0 or 1)
        double& mu_out,
        double& sigma_out
    ) {
        std::vector<double> regime_returns; // extract returns for the specific regime (0 or 1)
        
        for (size_t i = 0; i < returns.size(); ++i) {
            if (regime_sequence[i] == regime_id) {
                regime_returns.push_back(returns[i]);
            }
        }
        
        if (regime_returns.empty()) {
            // No data for this regime, return defaults
            mu_out = (regime_id == 0) ? 0.10 : -0.50;
            sigma_out = (regime_id == 0) ? 0.20 : 0.60;
            return;
        }               // Normal Distribution: rt​∼N(μi​/252,σi^2​/252)
        
        // Calculate mean (drift)  μi​ = 252 * E [rt ​∣ Zt​=i]
        double sum = std::accumulate(regime_returns.begin(), regime_returns.end(), 0.0);
        double mean_return = sum / regime_returns.size();
        mu_out = mean_return * 252.0;  // Daily to annual
        
        // Calculate SD (volatility) σi​ = sqrt(252)​ * sqrt[ Var(rt ​∣ Zt​=i) ]
        double var_sum = 0.0;
        for (double r : regime_returns) {
            double diff = r - mean_return;
            var_sum += diff * diff;
        }
        double variance = var_sum / regime_returns.size();
        sigma_out = std::sqrt(variance) * std::sqrt(252.0); // Annualized
    }

    // Estimate transition probabilities from regime sequence
    void estimateTransitionProbs(
        const std::vector<int>& regime_sequence,
        double& p_nc_out,
        double& p_cn_out
    ) {
        if (regime_sequence.size() < 2) {
            p_nc_out = 0.01;
            p_cn_out = 0.20;
            return;
        }
        
        int n_normal = 0, n_crash = 0;
        int transitions_nc = 0, transitions_cn = 0;
        
        for (size_t i = 0; i < regime_sequence.size() - 1; ++i) {
            if (regime_sequence[i] == 0) {  // Currently in Normal
                n_normal++;
                if (regime_sequence[i+1] == 1) {  // Transition to Crash
                    transitions_nc++;
                }
            } else {  // Currently in Crash
                n_crash++;
                if (regime_sequence[i+1] == 0) {  // Transition to Normal
                    transitions_cn++;
                }
            }
        }
        // Calculate probabilities
        p_nc_out = (n_normal > 0) ? static_cast<double>(transitions_nc) / n_normal : 0.01;
        p_cn_out = (n_crash > 0) ? static_cast<double>(transitions_cn) / n_crash : 0.20;
        
        // Ensure probabilities are in reasonable range [0.001, 0.5] for p_nc, [0.001, 0.99] for p_cn
        p_nc_out = std::max(0.001, std::min(0.5, p_nc_out));
        p_cn_out = std::max(0.001, std::min(0.99, p_cn_out));
    }
}


// -----------------------------Implementation of RegimeConfig::calibrateFromData-----------------------------------------

RegimeConfig RegimeConfig::calibrateFromData(
    const MarketData& data,
    int rolling_window,
    bool use_clustering
) {
    // Validate input data
    if (!data.isValid()) {
        throw std::invalid_argument("Invalid market data: need at least 2 price points");
    }
    if (data.size() < 20) {
        throw std::invalid_argument("Need at least 20 price points for calibration");
    }
    
    std::cout << "\n===== Calibrating Regime-Switching Parameters =====\n";
    std::cout << "  Data points: " << data.size() << "\n";
    std::cout << "  Rolling window: " << rolling_window << " days\n";
    
    // Step 1: Compute log returns
    auto returns = ParameterEstimator::computeLogReturns(data.prices);
    
    // Step 2: Compute rolling volatility
    auto volatilities = computeRollingVolatility(returns, rolling_window);
    
    // Step 3: Identify regimes (0=Normal, 1=Crash)
    std::vector<int> regime_sequence;
    regime_sequence = kMeansClustering(volatilities);
    
    // Step 4: Estimate parameters for each regime
    double normal_mu, normal_sigma, crash_mu, crash_sigma;
    estimateRegimeParams(returns, regime_sequence, 0, normal_mu, normal_sigma);
    estimateRegimeParams(returns, regime_sequence, 1, crash_mu, crash_sigma);
    
    // Step 5: Estimate transition probabilities
    double p_nc, p_cn;
    estimateTransitionProbs(regime_sequence, p_nc, p_cn);
    
    // Count regime occurrences for diagnostics
    int normal_count = std::count(regime_sequence.begin(), regime_sequence.end(), 0);
    int crash_count = regime_sequence.size() - normal_count;
    
    std::cout << "\n  Identified regimes:\n";
    std::cout << "    Normal periods: " << normal_count << " (" 
              << (100.0 * normal_count / regime_sequence.size()) << "%)\n";
    std::cout << "    Crash periods:  " << crash_count << " (" 
              << (100.0 * crash_count / regime_sequence.size()) << "%)\n";
    
    std::cout << "\n  Estimated parameters:\n";
    std::cout << "    Normal: μ = " << std::fixed << std::setprecision(3) << normal_mu 
              << ", σ = " << normal_sigma << "\n";
    std::cout << "    Crash: μ = " << crash_mu << ", σ = " << crash_sigma << "\n";
    std::cout << "    P(N→C) = " << p_nc << ", P(C→N) = " << p_cn << "\n";
    
    // Step 6: Return calibrated config
    return RegimeConfig(normal_mu, normal_sigma, crash_mu, crash_sigma, p_nc, p_cn);
}


// Constructor from config struct
RegimeSwitching::RegimeSwitching(
    const RegimeConfig& cfg,
    Regime initial_regime,
    unsigned int seed
) : current_regime(initial_regime),
    config(cfg),
    rng(seed),
    uniform_dist(0.0, 1.0),
    total_steps(0)
{
    buildTransitionMatrix();
    resetStatistics();
}

// Constructor with individual parameters
RegimeSwitching::RegimeSwitching(
    double normal_mu,
    double normal_sigma,
    double crash_mu,
    double crash_sigma,
    double normal_to_crash_prob,
    double crash_to_normal_prob,
    Regime initial_regime,
    unsigned int seed
) : RegimeSwitching(
        RegimeConfig(normal_mu, normal_sigma, crash_mu, crash_sigma, 
                     normal_to_crash_prob, crash_to_normal_prob),
        initial_regime,
        seed
    ) {}

void RegimeSwitching::buildTransitionMatrix() {
    // Row 0: From NORMAL
    transition_matrix[0][0] = 1.0 - config.normal_to_crash_prob;  // Stay in Normal
    transition_matrix[0][1] = config.normal_to_crash_prob;        // Normal -> Crash
    // Row 1: From CRASH
    transition_matrix[1][0] = config.crash_to_normal_prob;        // Crash -> Normal
    transition_matrix[1][1] = 1.0 - config.crash_to_normal_prob;  // Stay in Crash
}

void RegimeSwitching::updateRegime() {
    double rand_val = uniform_dist(rng);
    int current_state = static_cast<int>(current_regime);
    
    // Record current regime before potential transition
    regime_counts[current_state]++;
    
    // Determine if we transition to the other regime
    int other_state = 1 - current_state;  // 0 or 1
    
    if (rand_val < transition_matrix[current_state][other_state]) {
        // random value within the probablity -- Transition occurs
        transition_counts[current_state][other_state]++;
        current_regime = static_cast<Regime>(other_state);
    } else {
        // Stay in current regime
        transition_counts[current_state][current_state]++;
    }
    
    total_steps++;
}

RegimeParameters RegimeSwitching::getCurrentParameters() const {
    return (current_regime == Regime::NORMAL) 
           ? config.normal_params 
           : config.crash_params;
}

Regime RegimeSwitching::getCurrentRegime() const {
    return current_regime;
}

void RegimeSwitching::setConfig(const RegimeConfig& cfg) {
    config = cfg;
    buildTransitionMatrix();
}

RegimeConfig RegimeSwitching::getConfig() const {
    return config;
}

void RegimeSwitching::reset(Regime regime) {
    current_regime = regime;
    resetStatistics();
}

void RegimeSwitching::setCurrentRegime(Regime regime) {
    current_regime = regime;
}

void RegimeSwitching::resetStatistics() {
    for (int i = 0; i < 2; i++) {
        regime_counts[i] = 0;
        for (int j = 0; j < 2; j++) {
            transition_counts[i][j] = 0;
        }
    }
    total_steps = 0;
}

void RegimeSwitching::printTransitionMatrix() const {
    std::cout << "\n║          Transition Probability Matrix              ║\n";
    std::cout << "║                   To Normal    To Crash             ║\n";
    std::cout << "║  From Normal:       " << std::fixed << std::setprecision(2) 
              << std::setw(6) << transition_matrix[0][0] << "      " 
              << std::setw(6) << transition_matrix[0][1] << "              ║\n";
    std::cout << "║  From Crash:        " 
              << std::setw(6) << transition_matrix[1][0] << "      " 
              << std::setw(6) << transition_matrix[1][1] << "              ║\n";
}

void RegimeSwitching::printCurrentState() const {
    std::cout << "\n║               Current Regime State                   ║\n";
    
    RegimeParameters params = getCurrentParameters();
    std::cout << "║  Regime: " << std::setw(42) << std::left << params.name << "║\n";
    std::cout << "║  Drift (μ): " << std::fixed << std::setprecision(3) << std::setw(35) << std::right << params.mu << " ║\n";
    std::cout << "║  Volatility (σ): " << std::setw(35) << params.sigma << " ║\n";
}

void RegimeSwitching::printStatistics() const {
    if (total_steps == 0) {
        std::cout << "\nNo statistics available yet (no steps taken).\n";
        return;
    }

    std::cout << "\n║            Regime-Switching Statistics               ║\n";
    std::cout << "║  Total Steps: " << std::setw(38) << total_steps << " ║\n";
    
    // Empirical distribution
    auto [emp_normal, emp_crash] = getEmpiricalDistribution();
    std::cout << "║  Empirical Distribution:                             ║\n";
    std::cout << "║    Normal: " << std::fixed << std::setprecision(3) << std::setw(41) << emp_normal << " ║\n";
    std::cout << "║    Crash:  " << std::setw(41) << emp_crash << " ║\n";
    
    // Theoretical distribution
    auto [theo_normal, theo_crash] = getStationaryDistribution();
    std::cout << "║  Theoretical (Stationary):                           ║\n";
    std::cout << "║    Normal: " << std::setw(41) << theo_normal << " ║\n";
    std::cout << "║    Crash:  " << std::setw(41) << theo_crash << " ║\n";
    
    // Transition counts
    std::cout << "\n║  Transition Counts:                                ║\n";
    std::cout << "║    Normal → Normal: " << std::setw(30) << transition_counts[0][0] << " ║\n";
    std::cout << "║    Normal → Crash:  " << std::setw(30) << transition_counts[0][1] << " ║\n";
    std::cout << "║    Crash → Normal:  " << std::setw(30) << transition_counts[1][0] << " ║\n";
    std::cout << "║    Crash → Crash:   " << std::setw(30) << transition_counts[1][1] << " ║\n";
}

std::pair<double, double> RegimeSwitching::getStationaryDistribution() const {
    /* π = π * P
    [πN​,πC​] = [πN​,πC​] *（Pnn Pnc)
                        (Pcn Pcc) */
    
    // πN ​= πN * ​PNN ​+ πC * ​PCN​
    // πN ​= πN * ​PNC ​+ πC * ​PCC
    // πN + πC ​= 1

    // Solve for the above equation set: for 2-state Markov chain: π = [p_cn / (p_nc + p_cn), p_nc / (p_nc + p_cn)]
    double p_nc = transition_matrix[0][1];  // Normal → Crash
    double p_cn = transition_matrix[1][0];  // Crash → Normal
    
    double sum = p_nc + p_cn;
    if (sum < 1e-10) {  // Avoid division by zero
        return {0.5, 0.5};
    }
    
    double pi_normal = p_cn / sum;
    double pi_crash = p_nc / sum;
    
    return {pi_normal, pi_crash};
}

std::pair<double, double> RegimeSwitching::getExpectedDurations() const {
    // Expected duration = 1 / (prob of leaving state)
    double duration_normal = 1.0 / transition_matrix[0][1];  // 1 / P(Normal→Crash)
    double duration_crash = 1.0 / transition_matrix[1][0];   // 1 / P(Crash→Normal)
    
    return {duration_normal, duration_crash};
}

std::pair<double, double> RegimeSwitching::getEmpiricalDistribution() const {
    if (total_steps == 0) {
        return {0.0, 0.0};
    }
    
    double emp_normal = static_cast<double>(regime_counts[0]) / total_steps;
    double emp_crash = static_cast<double>(regime_counts[1]) / total_steps;
    
    return {emp_normal, emp_crash};
}

// CSV data loader implementation
namespace RegimeDataLoader {
    MarketData loadFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::vector<double> prices;
        std::string line;
        
        // Skip header if present (check if first line contains "price" or "Price")
        std::getline(file, line);
        if (line.find("price") == std::string::npos && 
            line.find("Price") == std::string::npos) {
            // No header, parse this line
            std::istringstream iss(line);
            std::string date, price_str;
            if (std::getline(iss, date, ',') && std::getline(iss, price_str)) {
                prices.push_back(std::stod(price_str));
            }
        }
        
        // Read data lines (format: date,price)
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string date, price_str;
            
            if (std::getline(iss, date, ',') && std::getline(iss, price_str)) {
                prices.push_back(std::stod(price_str));
            }
        }
        
        if (prices.empty()) {
            throw std::runtime_error("No valid price data found in file: " + filename);
        }
        
        std::cout << "Loaded " << prices.size() << " prices from " << filename << "\n";
        
        return MarketData(prices);
    }
}
