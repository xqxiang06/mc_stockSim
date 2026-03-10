#ifndef CORRELATION_H
#define CORRELATION_H

#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>

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
     * 2x2 correlation matrix (for stock-bond portfolio)
     * [ 1.0   ρ   ]        W1 & W2's own variance is 1
     * [  ρ   1.0  ]        Cov(W1, W2) = ρ
     */                     
    explicit CorrelationMatrix(double rho);
    
    /**
     * Generate a pair of correlated standard normals
     * @param Z1 Independent N(0,1) #1
     * @param Z2 Independent N(0,1) #2
     * @return [W1, W2] correlated normals
     */
    std::array<double, 2> generateCorrelated(double Z1, double Z2) const;
    
    /**
     * Get the correlation coefficient
     */
    double getCorrelation() const { return rho; }
    
    /**
     * Get the Cholesky decomposition matrix L (lower triangular)
     * Stored in row-major order: [L00, L10, L11]
     */
    const std::array<double, 3>& getCholeskyL() const { return L; }
    
private:
    double rho;  // Correlation coefficient [-1, 1]
    
    // Cholesky decomposition for lower triangle L: such that Σ = L*L^T
    // For 2x2: L = [ a   0  ]    where a = 1, b = ρ, c = √(1-ρ²)
    //              [ b   c  ]
    std::array<double, 3> L;  // [L00=a, L10=b, L11=c]
    
    void computeCholesky();
    void validateCorrelation() const;
};


/**
 * Estimate correlation from historical log returns of two assets
 */
class CorrelationEstimator {
public:
    /**
     * Estimate Pearson correlation between two return series
     * @param returns1 Log returns of asset 1
     * @param returns2 Log returns of asset 2
     * @return Correlation coefficient ∈ [-1, 1]
     */
    static double estimateCorrelation(
        const std::vector<double>& returns1,
        const std::vector<double>& returns2
    );
    
private:
    static double mean(const std::vector<double>& data);
    static double covariance(const std::vector<double>& x, const std::vector<double>& y);
    static double stddev(const std::vector<double>& data);
};

#endif