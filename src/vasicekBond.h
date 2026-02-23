#ifndef VASICEK_BOND_H
#define VASICEK_BOND_H
#include <vector>
#include <cmath>
#include <random>

/**
 * Vasicek Interest Rate Model
 * 
 * Stochastic differential equation:
 *   dr = κ(θ - r)dt + σ dW
 * 
 * Parameters:
 *   r(t) = short rate at time t
 *   κ    = mean reversion speed
 *   θ    = long-term mean rate (long term converges to this average level)
 *   σ    = volatility of interest rate
 */

struct VasicekParameters {
    double kappa;      // Mean reversion speed
    double theta;      // Long-term mean rate
    double sigma;      // Interest rate volatility
    double r0;         // Initial short rate
    
    VasicekParameters(double k, double t, double s, double r)
        : kappa(k), theta(t), sigma(s), r0(r) {}
};


class VasicekBond {
public:
    /**
     * Constructor
     * @param params Vasicek model parameters
     * @param T Total simulation time (years)
     * @param n_steps Number of time steps
     * @param maturity Bond maturity (years)
     */
    VasicekBond(const VasicekParameters& params, 
                double T, int n_steps,
                double maturity = 10.0);
    
    /**
     * Simulate a single path of short rates and bond prices
     * @param correlated_normal Pre-generated correlated normal for this path (if nullptr, generates independently)
     * @return Vector of bond prices at each time step
     */
    std::vector<double> simulatePath(const std::vector<double>* correlated_normals = nullptr);
    
    /**
     * Get current short rate (after simulation)
     */
    double getCurrentRate() const { return current_rate; }
    
    /**
     * Analytical bond price under Vasicek model
     * P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
     */
    double bondPrice(double r, double time_to_maturity) const;
    
    /**
     * Get the initial bond price
     */
    double getInitialPrice() const;
    
private:
    VasicekParameters params;
    double T;              // Total time
    int n_steps;           // Number of steps
    double dt;             // Time step size
    double maturity;       // Bond maturity in years
    double current_rate;   // Current short rate (updated during simulation)
    
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
    
    // Helper functions for analytical bond pricing
    double B(double tau) const;  // tau = time to maturity
    double A(double tau) const;
};


/**
 * Estimate Vasicek parameters from historical interest rate data
 */
class VasicekEstimator {
public:
    /**
     * Estimate parameters using maximum likelihood on discrete observations
     * @param rates Historical short rate observations
     * @param dt Time between observations (e.g., 1/252 for daily data)
     * @return Estimated Vasicek parameters
     */
    static VasicekParameters estimateFromRates(
        const std::vector<double>& rates,
        double dt = 1.0/252.0
    );
    
private:
    static double mean(const std::vector<double>& data);
    static double variance(const std::vector<double>& data);
    static double autocovariance(const std::vector<double>& data, int lag);
};

#endif