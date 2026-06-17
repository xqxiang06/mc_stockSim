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
    : mu(mu), cov(cov), rf(rf) {

    }

SharpeResult SharpeOptimizer::maxSharpe() const {
    // start the embedded Python interpreter
    py::scoped_interpreter guard{};

    // add src/ to sys.path so Python finds optimize_sharpe.py
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, ".");      // project root
    sys.attr("path").attr("insert")(0, "./src");  // src/ folder

    // import optimize_sharpe.py as a module
    py::module_ opt = py::module_::import("optimalShar");

    // convert std::array to std::vector — pybind11 maps vector → Python list
    std::vector<double> mu_vec(mu.begin(), mu.end());

    // call max_sharpe(mu, cov, rf) — cov is already vector<vector<double>>
    py::dict result = opt.attr("max_sharpe")(mu_vec, cov, rf);

    // read results back from Python dict into C++ struct
    std::vector<double> w = result["weights"].cast<std::vector<double>>();

    SharpeResult r;
    r.weights         = { w[0], w[1], w[2] };
    r.expected_return = result["ret"].cast<double>();
    r.volatility      = result["vol"].cast<double>();
    r.sharpe          = result["sharpe"].cast<double>();

    return r;
}

void SharpeOptimizer::printResult(const SharpeResult &r) const{
    const char* labels[] = {"VOO", "VXUS", "BND"};
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Max Sharpe Portfolio (scipy SLSQP) ===\n";

    for (int i = 0; i < 3; ++i)
        std::cout << " " << labels[i] << ": "
                  << r.weights[i]*100 << "%\n";
    
    std::cout << "  Expected return: " << r.expected_return * 100 << "%\n";
    std::cout << "  Volatility:      " << r.volatility * 100 << "%\n";
    std::cout << "  Sharpe ratio:    " << r.sharpe << "\n";
}