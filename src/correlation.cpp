#include "correlation.h"
#include <cmath>
#include <numeric>
#include <algorithm>

CorrelationMatrix::CorrelationMatrix(double rho) : rho(rho) {
    validateCorrelation();
    computeCholesky();
}

void CorrelationMatrix::validateCorrelation() const {
    if (rho < -1.0 || rho > 1.0) {
        throw std::invalid_argument("Correlation must be in [-1, 1]");
    }
}

void CorrelationMatrix::computeCholesky() {
    // For 2x2 correlation matrix:
    // Σ = [ 1   ρ ]
    //     [ ρ   1 ]
    //
    // Cholesky: L = [ a   0 ]    where a=1, b=ρ, c=√(1-ρ²)
    //               [ b   c ]
    //
    // Verify: L*L^T = [ a²      ab    ] = [ 1   ρ ]
    //                 [ ab     b²+c²  ]   [ ρ   1 ]
    
    L[0] = 1.0;                           // L00 = a
    L[1] = rho;                           // L10 = b
    L[2] = std::sqrt(1.0 - rho * rho);    // L11 = c = √(1-ρ²)
}

std::array<double, 2> CorrelationMatrix::generateCorrelated(double Z1, double Z2) const {
    // W = L * Z
    // W1 = L00*Z1 + 0*Z2   = Z1
    // W2 = L10*Z1 + L11*Z2 = ρ*Z1 + √(1-ρ²)*Z2
    
    double W1 = L[0] * Z1;                // = Z1
    double W2 = L[1] * Z1 + L[2] * Z2;    // = ρ*Z1 + √(1-ρ²)*Z2
    
    return {W1, W2};
}


// ==================== CorrelationEstimator Implementation ====================

double CorrelationEstimator::estimateCorrelation(
    const std::vector<double>& returns1,
    const std::vector<double>& returns2)
{
    if (returns1.size() != returns2.size()) {
        throw std::invalid_argument("Return series must have same length");
    }
    if (returns1.size() < 2) {
        throw std::invalid_argument("Need at least 2 observations");
    }
    
    // Pearson correlation: ρ = Cov(X,Y) / (σ_X * σ_Y)
    double cov = covariance(returns1, returns2);
    double std1 = stddev(returns1);
    double std2 = stddev(returns2);
    
    if (std1 < 1e-10 || std2 < 1e-10) {
        return 0.0;  // One series is constant
    }
    
    double corr = cov / (std1 * std2);
    
    // Clamp to [-1, 1] to handle numerical errors
    return std::max(-1.0, std::min(1.0, corr));
}

double CorrelationEstimator::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double CorrelationEstimator::covariance(
    const std::vector<double>& x,
    const std::vector<double>& y)
{
    if (x.size() != y.size() || x.size() < 2) return 0.0;
    
    double mean_x = mean(x);
    double mean_y = mean(y);
    
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    
    return sum / (x.size() - 1);  // Sample covariance
}

double CorrelationEstimator::stddev(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    
    double m = mean(data);
    double sq_sum = 0.0;
    for (double x : data) {
        sq_sum += (x - m) * (x - m);
    }
    
    return std::sqrt(sq_sum / (data.size() - 1));
}