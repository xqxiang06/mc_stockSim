#include "montecarlo_gbm.h"
#include <algorithm>
#include <numeric>
#include <utility>
#include <chrono>
#include <stdexcept>
#include <iostream>
// For parallel
#include <execution>
#include <thread>

/* log-price with jump: dlnSt​=(μ − 0.5*​σ^2 − λκ)dt + σdBt ​+ JdNt​
    Jump size: J ~ N(mu_J, sigma_J^2)
    where κ = E[e^J − 1]                                      */

//-----------------------------------------Class-MonteCarloGBM--------------------------------------------

MonteCarloGBM::MonteCarloGBM(double S0, double mu, double sigma,
                             double T, int n_steps, int n_paths,
                             double lambda, double mu_J, double sigma_J,
                             bool risk_neutral, double r)
    : S0(S0), mu(mu), sigma(sigma), T(T), n_steps(n_steps), n_paths(n_paths),
      lambda(lambda), mu_J(mu_J), sigma_J(sigma_J),
      risk_neutral(risk_neutral), r(r)
{
    generateTimeGrid();

    // Pre-allocate paths matrix: n_paths x (n_steps+1)
    paths.resize(n_paths, std::vector<double>(n_steps + 1));
}

void MonteCarloGBM::generateTimeGrid()
{
    time_grid.resize(n_steps + 1);
    for (int i = 0; i <= n_steps; ++i)
    {
        time_grid[i] = (i * T) / n_steps; // dt for each step (cumulative)
    }
}

// Thread-local uniform random number generator [0,1)
static inline double thread_local_uniform()
{
    thread_local struct
    {
        uint64_t state = std::hash<std::thread::id>{}(std::this_thread::get_id()) + 0x853c49e6748fea9bULL;
        uint64_t inc = (std::hash<std::thread::id>{}(std::this_thread::get_id()) << 1u) | 1u;
        uint32_t next()
        {
            uint64_t oldstate = state;
            state = oldstate * 6364136223846793005ULL + inc;
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }
    } rng;
    return rng.next() * 2.3283064365386963e-10;
}

// Thread-local normal random number generator (based on uniform)
static inline double thread_local_normal()
{
    double u1 = thread_local_uniform();
    double u2 = thread_local_uniform();
    if (u1 < 1e-10f)
        u1 = 1e-10f;
    double r = std::sqrt(-2.0f * std::log(u1));
    double theta = 6.28318530718f * u2;
    return r * std::cos(theta);
}

void MonteCarloGBM::simulate()
{
    std::cout << "\n===== Monte Carlo Simulation Parameters =====\n"
              << "  S0 = $" << S0 << std::endl
              << "  mu = " << mu << " (annual return)\n"
              << "  sigma = " << sigma << " (annual volatility)\n"
              << "  T = " << T << " years\n"
              << "  Steps = " << n_steps << ", Paths = " << n_paths << std::endl;
    // ADD: Display jump parameters if enabled
    if (hasJumps()) {
        std::cout << "  [Jump Diffusion Enabled]\n"
                  << "  lambda = " << lambda << " (jumps/year)\n"
                  << "  mu_J = " << mu_J << " (mean jump)\n"
                  << "  sigma_J = " << sigma_J << " (jump volatility)\n";
    }

    std::cout << "  Total random numbers: " << (n_steps * n_paths) << std::endl;

    const double dt = T / n_steps;
    const double sqrt_dt = std::sqrt(dt);
    double drift_rate = risk_neutral ? r : mu; // Use r if risk-neutral
    double drift = drift_rate - 0.5 * sigma * sigma;
    // ADD: Drift with jump compensation
    if (hasJumps()) {
        double kappa = std::exp(mu_J + 0.5 * sigma_J * sigma_J) - 1.0; // expected percentage change
        drift -= lambda * kappa;  // Add jump drift correction
        
        // Assume no jump premium for jump distribution（P=Q for jumps）
    }

    paths.assign(n_paths, std::vector<double>(n_steps + 1));
    final_prices.assign(n_paths, 0.0); // pre-allocate


    // std::for_each(std::execution::par, this->paths.begin(), this->paths.end(), [&](std::vector<double>& path) {
    for (std::vector<double> &path : paths) {
        // Set initial prices
        path[0] = S0;
        double log_S = std::log(S0); // use incremental form

        for (int i = 1; i <= n_steps; ++i) {
            // Brownian Increment
            double dW = thread_local_normal() * sqrt_dt; // dW ~ N(0, √dt)
        
            // calculate log price increment (lognormal distribution)
            // d(log S) = (μ - σ²/2 - λκ)dt + σ dW + Jumps
            double d_log_S = drift * dt + sigma * dW;

            // ADD: Jump component (if enabled)
            if (hasJumps()) {
                // Poisson: approximate number of jumps in time step dt
                double expected_jumps = lambda * dt; // Ni​∼Poisson(λΔt)
                int num_jumps = 0;
                
                // Simple Poisson generation
                if (expected_jumps > 0) {
                    double L = std::exp(-expected_jumps); // P(N=0) = e^(−λΔt): the prob of NO events occurring in dt
                    double p = 1.0; // initialize the product variable
                    while (p > L) {
                        p *= (thread_local_uniform()); // p = U1 ​× U2 ​× ⋯
                        num_jumps++;
                    }
                    num_jumps--; // The last time in the loop Just crossed the threshold
                }                // actual number of jumps should be reduced by 1

                // Add jump effects to the log price increment
                for (int j = 0; j < num_jumps; ++j) {
                    double jump_size = mu_J + sigma_J * thread_local_normal(); // Jk ​∼ N(μ_J, σ_J²)
                    d_log_S += jump_size; // Add jump to this step's increment
                }
            }
            
            log_S += d_log_S; // smooth + jump
            path[i] = std::exp(log_S);
        }
    };

    for (int i = 0; i < n_paths; ++i) {
        final_prices[i] = paths[i][n_steps]; // Store last time step for all paths
    }

    
    std::cout << "Sample initial prices: $";
    for (int i = 0; i < std::min(5, n_paths); ++i)
    {
        std::cout << paths[i][0] << " ";
    }
    std::cout << std::endl;

    std::cout << "Sample final prices: $";
    for (int i = 0; i < std::min(5, n_paths); ++i)
    { // show 5 sample paths
        std::cout << paths[i][n_steps] << " ";
    }
    std::cout << std::endl;
}

const std::vector<double>& MonteCarloGBM::getFinalPrices() const
{
    return final_prices;
}   // Return last time step for all paths

double MonteCarloGBM::getMeanFinalPrice() const
{
    const auto& final_prices = getFinalPrices();
    double sum = std::accumulate(final_prices.begin(), final_prices.end(), 0.0);
    return sum / n_paths;
}

double MonteCarloGBM::getMedianFinalPrice() const
{
    return percentile(getFinalPrices(), 0.5);
} 

std::pair<double, double> MonteCarloGBM::getConfidenceInterval(double confidence) const
{

    const auto& final_prices = getFinalPrices();
    double lower_p = (1 - confidence) * 0.5;
    double upper_p = 1 - lower_p;

    double lower = percentile(final_prices, lower_p);
    double upper = percentile(final_prices, upper_p);

    return {lower, upper};
}

double MonteCarloGBM::percentile(const std::vector<double> &data, double p)
{
    if (data.empty())
        throw std::invalid_argument("Cannot compute percentile of empty data");
    if (p < 0 || p > 1)
        throw std::invalid_argument("Percentile must be between 0 and 1");

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    double index = p * (sorted_data.size() - 1);
    int lower_idx = static_cast<int>(std::floor(index)); // eg. 2.7~2
    int upper_idx = static_cast<int>(std::ceil(index));  // eg. 2.7~3

    if (lower_idx == upper_idx)
    {
        return sorted_data[lower_idx];
    }

    // Linear interpolation: Q(p) = (1-w) * Xlower + w * Xupper
    double weight = index - lower_idx;
    return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight;
}

//--------------------------------------------------Class-ParameterEstimator--------------------------------------------

std::pair<double, double> ParameterEstimator::estimateFromPrices(
    const std::vector<double> &prices, int trading_days_per_year)
{
    // Compute log returns
    std::vector<double> log_returns = computeLogReturns(prices);

    // Estimate mu: annualized mean return
    double mu_daily = mean(log_returns);
    double mu = mu_daily * trading_days_per_year;

    // Estimate sigma: annualized volatility
    double sigma_daily = stddev(log_returns);
    double sigma = sigma_daily * std::sqrt(trading_days_per_year);

    return {mu, sigma};
}


ParameterEstimator::JumpParameters ParameterEstimator::estimateJumpParameters(
    const std::vector<double>& prices,
    double threshold, int trading_days_per_year)
{
    // Compute log returns
    auto log_returns = computeLogReturns(prices);
    
    // Calculate mean and std dev of all returns
    double mu_daily = mean(log_returns);
    double sigma_daily = stddev(log_returns);
    
    // Separate returns into "jumps" and "normal moves"
    std::vector<double> jump_returns;
    std::vector<double> normal_returns;
    
    for (double r : log_returns) {
        // Standardized return (z-score) to distinguish Jump / Normal
        double z_score = std::abs((r - mu_daily) / sigma_daily); // z = |_ - mean| / std
        
        if (z_score > threshold) {
            // A jump - store deviation from mean
            jump_returns.push_back(r - mu_daily);
        } else {
            // Normal market movement
            normal_returns.push_back(r);
        }
    }
    
    // Estimate lambda (annual jump frequency)
    double lambda = 0.0;
    if (!log_returns.empty()) {
        double jump_rate_daily = static_cast<double>(jump_returns.size()) / log_returns.size();
        lambda = jump_rate_daily * trading_days_per_year;
    }
    
    // Estimate jump size parameters
    double mu_J = 0.0;
    double sigma_J = 0.0;
    if (!jump_returns.empty()) {
        mu_J = mean(jump_returns);
        sigma_J = stddev(jump_returns);
    }
    
    // Estimate "smooth" volatility (excluding jumps)
    double sigma_smooth = 0.0;
    if (!normal_returns.empty()) {
        double sigma_smooth_daily = stddev(normal_returns);
        sigma_smooth = sigma_smooth_daily * std::sqrt(trading_days_per_year);
    } else {
        // Fallback to original sigma if no normal returns
        sigma_smooth = sigma_daily * std::sqrt(trading_days_per_year);
    }
    
    return {lambda, mu_J, sigma_J, sigma_smooth};
}


std::vector<double> ParameterEstimator::computeLogReturns(const std::vector<double> &prices)
{
    std::vector<double> log_returns;
    log_returns.reserve(prices.size() - 1); // reserve size-1 spaces

    for (size_t i = 1; i < prices.size(); ++i)
    {
        log_returns.push_back(std::log(prices[i] / prices[i - 1]));
    } // LOG Return: rt = ln(St / St-1)
    return log_returns;
}

double ParameterEstimator::mean(const std::vector<double> &data)
{
    if (data.empty())
        return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double ParameterEstimator::stddev(const std::vector<double> &data)
{
    if (data.size() < 2)
        return 0.0;

    double m = mean(data);
    double sq_sum = 0.0;
    for (double x : data)
    {
        sq_sum += (x - m) * (x - m);
    }
    return std::sqrt(sq_sum / (data.size() - 1)); // Sample std dev
}
