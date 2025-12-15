#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Read Date + Adj Close, return vector of Adj Close prices
std::vector<double> readAdjCloseCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }

    std::vector<double> prices;
    std::string line;

    // 1. Skip header line
    std::getline(file, line);

    // 2. Read each data row
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;

        // Column 0: Date (skip)
        std::getline(ss, cell, ',');

        // Column 1: Adj Close (keep)
        std::getline(ss, cell, ',');

        try {
            prices.push_back(std::stod(cell));
        } catch (...) {
            continue;
        }
    }

    if (prices.empty()) {
        throw std::runtime_error("No price data read from CSV");
    }

    return prices;
}
