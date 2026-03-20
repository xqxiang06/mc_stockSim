#ifndef CORRELATION_H
#define CORRELATION_H

#include "montecarlo_gbm.h"

#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <tuple>

/**
 * Correlation utilities for multi-asset simulation
 * 
 * Generates correlated Brownian motions using Cholesky decomposition:
 *   Given correlation matrix Σ, find lower triangular L such that Σ = L*L^T
 *   Then: W = L*Z where Z ~ N(0,I) gives correlated normals with cov(W) = Σ
 */

class CorrelationMatrix {
public:
    /**
     * 3x3 correlation matrix (for 3 fund portfolio)
     * {1.0,  0.75, -0.2},  // US Stock:    self, international, bond
     * {0.75, 1.0,  -0.1},  // Intl Stock:  US, self, bond
     * {-0.2, -0.1,  1.0}   // Bond:        US, international, self
     * used for example
     */                     
    explicit CorrelationMatrix(const std::vector<std::vector<double>> &corr_matrix);
    
    /**
     * Generate three correlated standard normals from three independent ones
     * @param Z1 Independent N(0,1) #1
     * @param Z2 Independent N(0,1) #2
     * @param Z3                        = normal_dist(rng);
     * @return [W1, W2, W3] correlated normals
     */
    std::tuple<double, double, double> generateCorrelated(double Z1, double Z2, double Z3) const;
    
    /**
     * Get the correlation matrix
     */
    const std::vector<std::vector<double>>& getCorrelationMatrix() const { return corr_matrix; }

    void printCholeskyFactors() const; // for debugging
    
private:
    std::vector<std::vector<double>> corr_matrix;
    
    // Cholesky decomposition for lower triangle L: such that Σ = L*L^T
    // for 3 * 3
    // L = [ L11   0    0  ]
    //     [ L21  L22   0  ]
    //     [ L31  L32  L33 ]
    double L11, L21, L31, L22, L32, L33;
    
    void computeCholesky();
    void validateMatrix() const;
};


/**
 * Estimate correlation from historical log returns of two assets
 */
class CorrelationEstimator {
public:
    /**
     * Estimate the full 3x3 Pearson correlation matrix from three return series
     * @param returns1 Log returns of asset 1 (US stocks)
     * @param returns2 Log returns of asset 2 (Intl stocks)
     * @param returns3 Log returns of asset 3 (Bonds)
     * @return Symmetric 3x3 correlation matrix with 1s on diagonal
     */
    static std::vector<std::vector<double>> estimateMatrix(
        const std::vector<double>& returns1,
        const std::vector<double>& returns2,
        const std::vector<double>& returns3
    );

    // Take raw prices, handles log-return conversion + trimming internally
    static std::vector<std::vector<double>> estimateFromPrices(
        const std::vector<double>& prices1,
        const std::vector<double>& prices2,
        const std::vector<double>& prices3);

    // Prints the 3x3 matrix with a label
    static void printMatrix(const std::vector<std::vector<double>>& corr, size_t n_obs);
    
private:
    static double mean(const std::vector<double>& data);
    static double covariance(const std::vector<double>& x, const std::vector<double>& y);
    static double stddev(const std::vector<double>& data);
};

#endif