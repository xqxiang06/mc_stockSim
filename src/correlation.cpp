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
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (corr_matrix[i][j] < -1.0 || corr_matrix[i][j] > 1.0) {
                throw std::invalid_argument("Correlation must be in [-1, 1]");
            }
        }
    }

    if (corr_matrix.size() != 3) {
        throw std::invalid_argument("Correlation matrix must be 3x3");
    }

    for (int i = 0; i < 3; ++i) {
        if (std::abs(corr_matrix[i][i] - 1.0) > 1e-6) {
            throw std::invalid_argument("Diagonal elements must be 1.0");
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
    L22 = std::sqrt(1.0 - L21 * L21);
    // L32 = (ρ23 - L21*L31) / L22
    L32 = (rho23 - L21 * L31) / L22;
    
    // Third column of L
    // L33 = sqrt(1 - L31² - L32²)
    L33 = std::sqrt(1.0 - L31 * L31 - L32 * L32);

    printCholeskyFactors();
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