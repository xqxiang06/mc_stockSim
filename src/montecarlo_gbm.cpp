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

//-----------------------------------------Class-MonteCarloGBM--------------------------------------------

MonteCarloGBM::MonteCarloGBM(double S0, double mu, double sigma,
                             double T, int n_steps, int n_paths)
    : S0(S0), mu(mu), sigma(sigma), T(T), n_steps(n_steps), n_paths(n_paths),
      normal_dist(0.0, 1.0)
{
    // Seed RNG with current time
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(seed);

    generateTimeGrid();

    // Pre-allocate paths matrix: (n_steps+1) x n_paths
    paths.resize(n_paths, std::vector<double>(n_steps + 1));
}

void MonteCarloGBM::generateTimeGrid()
{
    time_grid.resize(n_steps + 1);
    for (int i = 0; i <= n_steps; ++i)
    {
        time_grid[i] = (i * T) / n_steps; // dt for each step
    }
}

static inline double thread_local_normal()
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
        float next_float() { return next() * 2.3283064365386963e-10f; }
    } rng; // Box-Muller transform
    double u1 = rng.next_float();
    double u2 = rng.next_float();
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
              << "  Steps = " << n_steps << ", Paths = " << n_paths << std::endl
              << "  Total random numbers: " << (n_steps * n_paths) << std::endl;

    const double dt = T / n_steps;
    const double sqrt_dt = std::sqrt(dt);
    const double drift = mu - 0.5 * sigma * sigma;

#if 0
    // Generate paths
    for (int path = 0; path < n_paths; ++path) {
        std::vector<double> W(n_steps + 1);
        W[0] = 0.0; //For Brownian Motion: W(0) = 0
        paths[path][0] = S0; // Set initial prices

        for (int i = 1; i <= n_steps; ++i) {
            double dW = normal_dist(rng) * sqrt_dt; // dW ~ N(0, √dt)
            W[i] = W[i-1] + dW;  // Cumulative sum of dW
            // GBM model: S(t) = S₀ × exp((μ - σ²/2)t + σW(t))
            paths[path][i] = S0 * std::exp(drift * time_grid[i] + sigma * W[i]);
        }
    }
#else
    std::for_each(std::execution::par, this->paths.begin(), this->paths.end(), [&](std::vector<double> &step) {
        std::vector<double> W(n_steps + 1);
        W[0] = 0.0; //For Brownian Motion: W(0) = 0
        step[0] = S0; // Set initial prices
        for (int i = 1; i <= n_steps; ++i) {
            double dW = thread_local_normal() * sqrt_dt; // dW ~ N(0, √dt)
            W[i] = W[i-1] + dW;  // Cumulative sum of dW
            // GBM model: S(t) = S₀ × exp((μ - σ²/2)t + σW(t))
            step[i] = S0 * std::exp(drift * time_grid[i] + sigma * W[i]);
        } 
    });
#endif

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

std::vector<double> MonteCarloGBM::getFinalPrices() const
{
    std::vector<double> finalPrices;
    finalPrices.reserve(n_paths);
    for (int i = 0; i < n_paths; ++i)
    {
        finalPrices.push_back(paths[i][n_steps]);
    }
    return finalPrices; // Return last time step for all paths
}

double MonteCarloGBM::getMeanFinalPrice() const
{
    const auto &final_prices = getFinalPrices();
    double sum = std::accumulate(final_prices.begin(), final_prices.end(), 0.0);
    return sum / n_paths;
}

double MonteCarloGBM::getMedianFinalPrice() const
{
    return percentile(getFinalPrices(), 0.5);
} // Final prices

std::pair<double, double> MonteCarloGBM::getConfidenceInterval(double confidence) const
{

    const auto &final_prices = getFinalPrices();
    double lower_p = (1 - confidence) * 0.5;
    double upper_p = 1 - lower_p;

    double lower = percentile(final_prices, lower_p);
    double upper = percentile(final_prices, upper_p);

    return {lower, upper};
}

double MonteCarloGBM::percentile(const std::vector<double> &data, double p) const
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
