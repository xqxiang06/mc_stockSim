#include "Optimizer.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace py = pybind11;

SharpeOptimizer::SharpeOptimizer(
    const std::array<double, 3>& mu,
    const std::vector<std::vector<double>>& cov,
    double rf)
    : mu(mu), cov(cov), rf(rf), guard{} {
                             // embedded Python interpreter
    }                        // initialize guard in constructor

SharpeResult SharpeOptimizer::runOptimizer(const std::string &mode) const {
    // add src/ to sys.path so Python finds optimize_sharpe.py
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, ".");      // project root
    sys.attr("path").attr("insert")(0, "./src");  // src/ folder

    // import optimize_sharpe.py as a module
    py::module_ opt = py::module_::import("optimalShar");

    // convert std::array to std::vector — pybind11 maps vector → Python list
    std::vector<double> mu_vec(mu.begin(), mu.end());

    // call python optomizer()
    py::dict result = opt.attr("optimizer")(mu_vec, cov, rf, mode);

    // read results back from Python dict into C++ struct
    std::vector<double> w = result["weights"].cast<std::vector<double>>();

    SharpeResult r;
    r.weights         = { w[0], w[1], w[2] };
    r.expected_return = result["ret"].cast<double>();
    r.volatility      = result["vol"].cast<double>();
    r.sharpe          = result["sharpe"].cast<double>();

    return r;
}

// Two modes: whether max Sharpe or min vol
SharpeResult SharpeOptimizer::maxSharpe() const{
    return runOptimizer("max_sharpe");
}

SharpeResult SharpeOptimizer::minVol() const {
    return runOptimizer("min_vol");
}

void SharpeOptimizer::printResult(const SharpeResult &r,
                                  const std::string &label) const{
    const char* labels[] = {"VOO", "VXUS", "BND"};
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== " << label << "===\n";

    for (int i = 0; i < 3; ++i)
        std::cout << " " << labels[i] << ": "
                  << r.weights[i]*100 << "%\n";
    
    std::cout << "  Expected return: " << r.expected_return * 100 << "%\n";
    std::cout << "  Volatility:      " << r.volatility * 100 << "%\n";
    std::cout << "  Sharpe ratio:    " << r.sharpe << "\n";
}