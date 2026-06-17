#include "correlation.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <iostream>

CorrelationMatrix::CorrelationMatrix(const std::vector<std::vector<double>> &corr_matrix)
    : corr_matrix(corr_matrix) 
{
    validateMatrix();
    computeCholesky();
}

void CorrelationMatrix::validateMatrix() const {
    if (corr_matrix.size() != 3) {
        throw std::invalid_argument("Correlation matrix must be 3x3");
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (corr_matrix[i][j] < -1.0 || corr_matrix[i][j] > 1.0) {
                throw std::invalid_argument("Correlation must be in [-1, 1]");
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (std::abs(corr_matrix[i][i] - 1.0) > 1e-6) {
            throw std::invalid_argument("Diagonal elements must be 1.0");
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (std::abs(corr_matrix[i][j] - corr_matrix[j][i]) > 1e-6) {
                throw std::invalid_argument("Correlation Matrix must be symmetric");
            }
        }
    }
}

void CorrelationMatrix::computeCholesky() {
    // Extract correlation coefficients
    double rho12 = corr_matrix[0][1];  // Correlation between asset 1 and 2
    double rho13 = corr_matrix[0][2];  // Correlation between asset 1 and 3
    double rho23 = corr_matrix[1][2];  // Correlation between asset 2 and 3
    
    // Cholesky decomposition for 3x3 correlation matrix:
    // Σ = [ 1    ρ12  ρ13 ]
    //     [ ρ12  1    ρ23 ]
    //     [ ρ13  ρ23  1   ]
    //
    // L = [ L11   0    0  ]
    //     [ L21  L22   0  ]
    //     [ L31  L32  L33 ]
    //
    // Such that Σ = L * L^T
    
    // First column of L
    L11 = 1.0;
    L21 = rho12;
    L31 = rho13;
    
    // Second column of L
    // L22 = sqrt(1 - L21²)
    double sqL22 = 1.0 - L21 * L21;
    if (sqL22 <= 1e-12) {
        throw std::invalid_argument("Correlation matrix is not positive definite (L22)");
    }
    L22 = std::sqrt(sqL22);
    
    // L32 = (ρ23 - L21*L31) / L22
    L32 = (rho23 - L21 * L31) / L22;
    
    // Third column of L
    // L33 = sqrt(1 - L31² - L32²)
    double sqL33 = 1.0 - L31 * L31 - L32 * L32;
    if (sqL33 <= 1e-12) {
        throw std::invalid_argument("Correlation matrix is not positive definite (L33)");
    }
    L33 = std::sqrt(sqL33);

    printCholeskyFactors(); // show for debugging
}

std::tuple<double, double, double> CorrelationMatrix::generateCorrelated(double Z1, double Z2, double Z3) const {
    // Transform independent standard normals Z1, Z2, Z3 to correlated standard normals W1, W2, W3
    // using W = L * Z
    
    double W1 = L11 * Z1;
    double W2 = L21 * Z1 + L22 * Z2;
    double W3 = L31 * Z1 + L32 * Z2 + L33 * Z3;

    return {W1, W2, W3};
}

void CorrelationMatrix::printCholeskyFactors() const {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Cholesky Lower Triangular Matrix L:\n";
    std::cout << "| " << std::setw(7) << L11 << "  " << std::setw(7) << 0.0 << "  " << std::setw(7) << 0.0 << " |\n";
    std::cout << "| " << std::setw(7) << L21 << "  " << std::setw(7) << L22 << "  " << std::setw(7) << 0.0 << " |\n";
    std::cout << "| " << std::setw(7) << L31 << "  " << std::setw(7) << L32 << "  " << std::setw(7) << L33 << " |\n";
    std::cout << std::defaultfloat;
}

// ==================== CorrelationEstimator Implementation ====================

EstimationResult CorrelationEstimator::estimateMatrix(
    const std::vector<double>& returns1,
    const std::vector<double>& returns2,
    const std::vector<double>& returns3)
{
    size_t n = returns1.size();
    if (n != returns2.size() || n != returns3.size()) {
        throw std::invalid_argument("Return series must have same length");
    }
    if (n < 2) {
        throw std::invalid_argument("Need at least 2 observations");
    }
    
    // Pass 1: compute means
    double mu0 = mean(returns1);
    double mu1 = mean(returns2);
    double mu2 = mean(returns3);

    // Pass 2: accumulate all 6 unique cross-products in one loop
    // Covariance matrix Sigma is symmetric, only need upper triangle:
    //   c00 = Var(US),        c11 = Var(INTL),       c22 = Var(BOND)
    //   c01 = Cov(US,INTL),   c02 = Cov(US,BOND),    c12 = Cov(INTL,BOND)
    double c00=0, c11=0, c22=0;
    double c01=0, c02=0, c12=0;

    for (size_t t = 0; t < n; ++t) {
        double d0 = returns1[t] - mu0;
        double d1 = returns2[t] - mu1;
        double d2 = returns3[t] - mu2;

        c00 += d0 * d0;
        c11 += d1 * d1;
        c22 += d2 * d2;
        c01 += d0 * d1;   // Cov(US, INTL)
        c02 += d0 * d2;   // Cov(US, BOND)
        c12 += d1 * d2;   // Cov(INTL, BOND)
    }

    // Divide by (n-1) for sample covariance
    double inv = 1.0 / static_cast<double>(n - 1);
    c00 *= inv;  c11 *= inv;  c22 *= inv;
    c01 *= inv;  c02 *= inv;  c12 *= inv;

    // Covariance Matrix for python to optimize portfilio weights
    std::vector<std::vector<double>> cov = {
        { c00, c01, c02 },
        { c01, c11, c12 },
        { c02, c12, c22 }
    }; // unscaled / per day

    // Annualized
    for (auto &row : cov)
        for (auto &val : row)
            val *= 252.0;

    // CorrMatrix: normalize Sigma -> R using rho_ij = Cov(i,j) / (sig_i * sig_j)
    double sig0 = std::sqrt(c00);
    double sig1 = std::sqrt(c11);
    double sig2 = std::sqrt(c22);

    auto safeCorr = [](double cov, double si, double sj) -> double {
        if (si < 1e-10 || sj < 1e-10) return 0.0;
        return std::max(-1.0, std::min(1.0, cov / (si * sj)));
    };

    std::vector<std::vector<double>> corr = {
        { 1.0,                        safeCorr(c01, sig0, sig1), safeCorr(c02, sig0, sig2) },
        { safeCorr(c01, sig0, sig1),  1.0,                       safeCorr(c12, sig1, sig2) },
        { safeCorr(c02, sig0, sig2),  safeCorr(c12, sig1, sig2), 1.0                       }
    };

    return { corr, cov };
}

EstimationResult CorrelationEstimator::estimateFromPrices(
    const std::vector<double>& prices1,
    const std::vector<double>& prices2,
    const std::vector<double>& prices3)
{                                       // from montecarlo_gbm.h
    auto returns1 = ParameterEstimator::computeLogReturns(prices1);
    auto returns2 = ParameterEstimator::computeLogReturns(prices2);
    auto returns3 = ParameterEstimator::computeLogReturns(prices3);

    // trim all three to shortest, they need have same length
    size_t n = std::min({returns1.size(), returns2.size(), returns3.size()});
    returns1.resize(n); returns2.resize(n); returns3.resize(n);

    auto result = estimateMatrix(returns1, returns2, returns3);
    printMatrix(result.corr, result.cov, n);
    return result;
}

void CorrelationEstimator::printMatrix(
    const std::vector<std::vector<double>> &corr,
    const std::vector<std::vector<double>>& cov,
    size_t n_obs)
{
    // correlation mareix
    std::cout << "\nEstimated Correlation Matrix (" << n_obs << " observations):\n";
    std::cout << std::fixed << std::setprecision(3);
    const char* labels[] = {"US  ", "INTL", "BOND"};
    for (int i = 0; i < 3; ++i) {
        std::cout << "  " << labels[i] << " [ ";
        for (double v : corr[i]) std::cout << std::setw(7) << v << " ";
        std::cout << "]\n";
    }

    // covariance matrix (new)
    std::cout << "\nCovariance Matrix (annualized):\n";
    const char* cov_labels[] = {"VOO ", "VXUS", "BND "};
    for (int i = 0; i < 3; ++i) {
        std::cout << "  " << cov_labels[i] << " [ ";
        for (double v : cov[i])
            std::cout << std::fixed << std::setprecision(6) << std::setw(10) << v << " ";
        std::cout << "]\n";
    }

    std::cout << std::defaultfloat;
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