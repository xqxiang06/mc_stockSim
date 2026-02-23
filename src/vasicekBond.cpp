#include "vasicekBond.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

VasicekBond::VasicekBond(const VasicekParameters& params,
                         double T, int n_steps,
                         double maturity)
    : params(params), T(T), n_steps(n_steps), 
      maturity(maturity), current_rate(params.r0),
      rng(std::random_device{}()), normal_dist(0.0, 1.0)
{
    dt = T / n_steps;
    
    if (params.kappa <= 0) {
        throw std::invalid_argument("Mean reversion speed (kappa) must be positive");
    }
    if (params.sigma <= 0) {
        throw std::invalid_argument("Volatility (sigma) must be positive");
    }
}

std::vector<double> VasicekBond::simulatePath(const std::vector<double>* correlated_normals) {
    std::vector<double> bond_prices;
    bond_prices.reserve(n_steps + 1);
    
    // Initial values
    double r = params.r0;
    current_rate = r;
    
    // Initial bond price
    bond_prices.push_back(bondPrice(r, maturity));
    
    // Simulate short rate path using Euler-Maruyama discretization
    // dr = κ(θ - r)dt + σ dW
    for (int i = 1; i <= n_steps; ++i) {
        double dW;
        if (correlated_normals != nullptr && i-1 < static_cast<int>(correlated_normals->size())) {
            // Use pre-generated correlated normal
            dW = (*correlated_normals)[i-1] * std::sqrt(dt);
        } else {
            // Generate independent normal
            dW = normal_dist(rng) * std::sqrt(dt);
        }
        
        // Euler-Maruyama step
        double drift = params.kappa * (params.theta - r) * dt;
        double diffusion = params.sigma * dW;
        
        r += drift + diffusion;
        
        // Ensure rate stays non-negative (practical constraint)
        r = std::max(r, 0.0);
        
        // Calculate time remaining to maturity
        double time_to_maturity = maturity - (i * dt);
        if (time_to_maturity < 0) {
            time_to_maturity = 0;  // Bond has matured
        }
        
        bond_prices.push_back(bondPrice(r, time_to_maturity));
    }
    
    current_rate = r;
    return bond_prices;
}
                            // tau: time to maturity
double VasicekBond::B(double tau) const {
    // B(τ) = (1 - e^(-κτ)) / κ
    if (params.kappa < 1e-10) {
        return tau;  // Limit as κ → 0
    }
    return (1.0 - std::exp(-params.kappa * tau)) / params.kappa;
}

double VasicekBond::A(double tau) const {
    // A(τ) = exp((θ - σ²/(2κ²))(B(τ) - τ) - σ²B(τ)²/(4κ))
    double B_tau = B(tau);
    double sigma_sq = params.sigma * params.sigma;
    double kappa_sq = params.kappa * params.kappa;
    
    double term1 = (params.theta - sigma_sq / (2.0 * kappa_sq)) * (B_tau - tau);
    double term2 = sigma_sq * B_tau * B_tau / (4.0 * params.kappa);
    
    return std::exp(term1 - term2);
}

double VasicekBond::bondPrice(double r, double time_to_maturity) const {
    if (time_to_maturity <= 0) {
        return 1.0;  // Bond at maturity pays $1
    }
    
    // P(t,T) = A(τ) * exp(-B(τ) * r(t))
    double A_tau = A(time_to_maturity);
    double B_tau = B(time_to_maturity);
    
    return A_tau * std::exp(-B_tau * r);
}

double VasicekBond::getInitialPrice() const {
    return bondPrice(params.r0, maturity);
}

// ==================== VasicekEstimator Implementation ====================

VasicekParameters VasicekEstimator::estimateFromRates(
    const std::vector<double>& rates,
    double dt)
{
    if (rates.size() < 3) {
        throw std::invalid_argument("Need at least 3 observations to estimate Vasicek parameters");
    }
    
    // Estimate using discrete-time approximation
    // r(t+dt) = r(t) + κ(θ - r(t))dt + σ√dt * ε
    // Which gives: r(t+dt) ≈ (1-κdt)r(t) + κθdt + σ√dt * ε
    
    // Compute lag-1 differences
    std::vector<double> dr;
    dr.reserve(rates.size() - 1);
    
    for (size_t i = 1; i < rates.size(); ++i) {
        dr.push_back(rates[i] - rates[i-1]);
    }
    
    // Estimate theta (long-term mean) as sample mean
    double theta_est = mean(rates);
    
    // Estimate kappa using autocorrelation
    double r_mean = mean(rates);
    double var_r = variance(rates);
    double acov1 = autocovariance(rates, 1);
    
    // From AR(1): ρ(1) = exp(-κdt), so κ = -ln(ρ(1))/dt
    double rho1 = (var_r > 1e-10) ? (acov1 / var_r) : 0.9;
    rho1 = std::max(0.01, std::min(0.99, rho1));  // Keep in reasonable range
    
    double kappa_est = -std::log(rho1) / dt;
    
    // Estimate sigma from residuals
    double var_dr = variance(dr);
    // Var(dr) ≈ σ²dt
    double sigma_est = std::sqrt(var_dr / dt);
    
    // Use last observed rate as r0
    double r0 = rates.back();
    
    // Apply reasonable bounds
    kappa_est = std::max(0.01, std::min(5.0, kappa_est));
    theta_est = std::max(0.001, std::min(0.20, theta_est));
    sigma_est = std::max(0.001, std::min(0.10, sigma_est));
    
    return VasicekParameters(kappa_est, theta_est, sigma_est, r0);
}

double VasicekEstimator::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double VasicekEstimator::variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    
    double m = mean(data);
    double sq_sum = 0.0;
    for (double x : data) {
        sq_sum += (x - m) * (x - m);
    }
    return sq_sum / (data.size() - 1);
}

double VasicekEstimator::autocovariance(const std::vector<double>& data, int lag) {
    if (data.size() <= static_cast<size_t>(lag)) return 0.0;
    
    double m = mean(data);
    double sum = 0.0;
    size_t n = data.size() - lag;
    
    for (size_t i = 0; i < n; ++i) {
        sum += (data[i] - m) * (data[i + lag] - m);
    }
    
    return sum / n;
}