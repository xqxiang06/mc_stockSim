#ifndef CSV_READER_H
#define CSV_READER_H

#include <vector>
#include <string>

// Simple CSV reader for stock price data
// Reads Adj Close column from Yahoo Finance CSV format

/**
 * Read Adj Close prices from CSV
 * Expected format: Date, Adj Close
 * 
 * @param filename Path to CSV file
 * @return Vector of Adj Close prices
 * @throws std::runtime_error if file cannot be opened or no data found
 */
std::vector<double> readAdjCloseCSV(const std::string& filename);

/**
 * Read last N Adj Close prices from CSV
 * 
 * @param n Number of selected prices to return
 */
std::vector<double> readLastNPrices(const std::string& filename, int n);

#endif
