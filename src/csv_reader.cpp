#include "csv_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

// Read Adj Close prices from CSV
// Expected format: Date, Adj Close

std::vector<double> readAdjCloseCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }
    
    std::vector<double> prices;
    std::string line; // store a single line
    
    // Skip header line
    std::getline(file, line); // reads a single line of text
    
    // Read each data row
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string cell; // store a single string
        
        // Column 0: Date (skip)
        std::getline(ss, cell, ',');
        
        // Column 1: Adj Close (keep)
        std::getline(ss, cell, ','); // read in data with sstream, "," as delimiter
        
        try {
            prices.push_back(std::stod(cell));
        } catch (...) {
            continue;
        }
    }
    
    if (prices.empty()) {
        throw std::runtime_error("No price data read from CSV: " + filename);
    }
    
    return prices;
}


// Read last N Adj Close prices from CSV
std::vector<double> readLastNPrices(const std::string& filename, int n) {
    // Read all prices first
    std::vector<double> all_prices = readAdjCloseCSV(filename);
    
    // If we have fewer prices than requested, return all
    if (all_prices.size() <= static_cast<size_t>(n)) {
        return all_prices;
    }
    // Return last N prices
    return std::vector<double>(
        all_prices.end() - n,
        all_prices.end()
    );
}
