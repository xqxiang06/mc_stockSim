#ifndef SHARPE_OPTIMIZER_H
#define SHARPE_OPTIMIZER_H
// #pragma once

#include <array>
#include <vector>
#include <string>

// result bundle: mirrors what python return
struct SharpeResult {
    std::array<double, 3> weights;
    double expected_return;
    double volatility;
    double sharpe;
};

class SharpeOptimizer {
public:
    SharpeOptimizer(
        const std::array<double, 3> &mu,
        const std::vector<std::vector<double>> &cov,
        double rf = 0.045
    );

    SharpeResult maxSharpe() const;
    void printResult(const SharpeResult &r) const;

private:
    std::array<double, 3> mu;
    std::vector<std::vector<double>> cov;
    double rf;
};

#endif