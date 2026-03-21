#ifndef REGIME_SWITCH_H
#define REGIME_SWITCH_H
#include <random>
#include <string>
#include <vector>
#include <utility>

// Enum for regime types
enum class Regime {
    NORMAL = 0,
    CRASH = 1
};

// Structure to hold parameters for a single regime
struct RegimeParameters {
    double mu;           // Annual drift rate
    double sigma;        // Annual volatility
    std::string name;    // Descriptive name
    
    // Constructors
    RegimeParameters() : mu(0.0), sigma(0.0), name("Unknown") {}
    RegimeParameters(double m, double s, const std::string& n) 
        : mu(m), sigma(s), name(n) {}
};

// Hold market data for calibration
struct MarketData {
    std::vector<double> prices;
    
    // Constructors
    MarketData() {}
    MarketData(const std::vector<double>& p) : prices(p) {}
    
    size_t size() const { return prices.size(); }
    bool isValid() const { return prices.size() > 1; }
};

// Struct to hold complete regime-switching configuration
struct RegimeConfig {
    RegimeParameters normal_params;
    RegimeParameters crash_params;
    double normal_to_crash_prob;
    double crash_to_normal_prob;
    
    // Constructor with defaults (manual settings)
    RegimeConfig(
        double normal_mu = 0.10,
        double normal_sigma = 0.20,
        double crash_mu = -0.50,
        double crash_sigma = 0.60,
        double p_nc = 0.01,
        double p_cn = 0.20
    ) : normal_params(normal_mu, normal_sigma, "Normal"),
        crash_params(crash_mu, crash_sigma, "Crash"),
        normal_to_crash_prob(p_nc),
        crash_to_normal_prob(p_cn) {}
    
    // Static method for calibration from data (Simple Volatility-Based)
    static RegimeConfig calibrateFromData(
        const MarketData &data,
        int rolling_window = 20
    );
};

class RegimeSwitching {
private:
    // Current state
    Regime current_regime;
    
    // Configuration
    RegimeConfig config;
    
    // Transition matrix [from][to]
    double transition_matrix[2][2];
    
    // Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
    // Statistics tracking (optional, for validation)
    int regime_counts[2];
    int transition_counts[2][2];
    int total_steps;
    
    // Helper to build transition matrix from config
    void buildTransitionMatrix();

    // Store simulation final prices
    std::vector<double> final_prices_regime;

public:
    // Constructor from config struct
    RegimeSwitching(
        const RegimeConfig& cfg,
        Regime initial_regime = Regime::NORMAL,
        unsigned int seed = std::random_device{}()
    );
    
    // Constructor with individual parameters
    RegimeSwitching(
        double normal_mu,
        double normal_sigma,
        double crash_mu,
        double crash_sigma,
        double normal_to_crash_prob,
        double crash_to_normal_prob,
        Regime initial_regime = Regime::NORMAL,
        unsigned int seed = std::random_device{}()
    );
    
    // Core functionality
    void updateRegime();  // Update regime based on Markov transition
    RegimeParameters getCurrentParameters() const;
    Regime getCurrentRegime() const;
    
    // Configuration management
    void setConfig(const RegimeConfig& cfg);
    RegimeConfig getConfig() const;
    
    // Utility functions
    void reset(Regime regime = Regime::NORMAL);
    void resetStatistics();
    void setCurrentRegime(Regime regime);
    
    // Analysis and validation
    void printTransitionMatrix() const;
    void printCurrentState() const;
    void printStatistics() const;
    std::pair<double, double> getStationaryDistribution() const;
    std::pair<double, double> getExpectedDurations() const;
    std::pair<double, double> getEmpiricalDistribution() const;
    
    // Getters for individual parameters
    double getNormalMu() const { return config.normal_params.mu; }
    double getNormalSigma() const { return config.normal_params.sigma; }
    double getCrashMu() const { return config.crash_params.mu; }
    double getCrashSigma() const { return config.crash_params.sigma; }

    // Regime simulation
    void simulate(double S0, double T, int n_steps, int n_paths);
    double getMeanFinalPrice() const;
    double getMedianFinalPrice() const;
    std::pair<double, double> getConfidenceInterval(double confidence = 0.95) const;
    const std::vector<double>& getFinalPrices() const { return final_prices_regime; }
    void writeResultsToCSV(const std::string& filename) const;
};

// Predefined configurations for common scenarios (examples)
namespace RegimePresets {
    // Typical market with rare crashes
    inline RegimeConfig Typical() {
        return RegimeConfig(0.10, 0.20, -0.50, 0.60, 0.01, 0.20);
    }
    
    // Frequent volatility (more boom-bust cycles)
    inline RegimeConfig Volatile() {
        return RegimeConfig(0.10, 0.20, -0.30, 0.45, 0.05, 0.40);
    }
    
    // Black swan events (rare but devastating)
    inline RegimeConfig BlackSwan() {
        return RegimeConfig(0.10, 0.20, -0.80, 0.90, 0.005, 0.10);
    }
    
    // Mild corrections (frequent, shallow)
    inline RegimeConfig MildCorrection() {
        return RegimeConfig(0.10, 0.20, -0.20, 0.35, 0.03, 0.50);
    }
}

// Optional utility for loading data from CSV
namespace RegimeDataLoader {
    // Load market data from CSV file (format: date,price)
    MarketData loadFromCSV(const std::string& filename);
}

#endif