#include "regime_switch.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Constructor from config struct
RegimeSwitching::RegimeSwitching(
    const RegimeConfig& cfg,
    Regime initial_regime,
    unsigned int seed
) : current_regime(initial_regime),
    config(cfg),
    rng(seed),
    uniform_dist(0.0, 1.0),
    total_steps(0)
{
    buildTransitionMatrix();
    resetStatistics();
}

// Constructor with individual parameters
RegimeSwitching::RegimeSwitching(
    double normal_mu,
    double normal_sigma,
    double crash_mu,
    double crash_sigma,
    double normal_to_crash_prob,
    double crash_to_normal_prob,
    Regime initial_regime,
    unsigned int seed
) : RegimeSwitching(
        RegimeConfig(normal_mu, normal_sigma, crash_mu, crash_sigma, 
                     normal_to_crash_prob, crash_to_normal_prob),
        initial_regime,
        seed
    ) {}

void RegimeSwitching::buildTransitionMatrix() {
    // Row 0: From NORMAL
    transition_matrix[0][0] = 1.0 - config.normal_to_crash_prob;  // Stay in Normal
    transition_matrix[0][1] = config.normal_to_crash_prob;        // Normal -> Crash
    // Row 1: From CRASH
    transition_matrix[1][0] = config.crash_to_normal_prob;        // Crash -> Normal
    transition_matrix[1][1] = 1.0 - config.crash_to_normal_prob;  // Stay in Crash
}

void RegimeSwitching::updateRegime() {
    double rand_val = uniform_dist(rng);
    int current_state = static_cast<int>(current_regime);
    
    // Record current regime before potential transition
    regime_counts[current_state]++;
    
    // Determine if we transition to the other regime
    int other_state = 1 - current_state;  // 0 or 1
    
    if (rand_val < transition_matrix[current_state][other_state]) {
        // random value within the probablity -- Transition occurs
        transition_counts[current_state][other_state]++;
        current_regime = static_cast<Regime>(other_state);
    } else {
        // Stay in current regime
        transition_counts[current_state][current_state]++;
    }
    
    total_steps++;
}

RegimeParameters RegimeSwitching::getCurrentParameters() const {
    return (current_regime == Regime::NORMAL) 
           ? config.normal_params 
           : config.crash_params;
}

Regime RegimeSwitching::getCurrentRegime() const {
    return current_regime;
}

void RegimeSwitching::setConfig(const RegimeConfig& cfg) {
    config = cfg;
    buildTransitionMatrix();
}

RegimeConfig RegimeSwitching::getConfig() const {
    return config;
}

void RegimeSwitching::reset(Regime regime) {
    current_regime = regime;
    resetStatistics();
}

void RegimeSwitching::setCurrentRegime(Regime regime) {
    current_regime = regime;
}

void RegimeSwitching::resetStatistics() {
    for (int i = 0; i < 2; i++) {
        regime_counts[i] = 0;
        for (int j = 0; j < 2; j++) {
            transition_counts[i][j] = 0;
        }
    }
    total_steps = 0;
}

void RegimeSwitching::printTransitionMatrix() const {
    std::cout << "\n║          Transition Probability Matrix              ║\n";
    std::cout << "║                   To Normal    To Crash             ║\n";
    std::cout << "║  From Normal:       " << std::fixed << std::setprecision(2) 
              << std::setw(6) << transition_matrix[0][0] << "      " 
              << std::setw(6) << transition_matrix[0][1] << "              ║\n";
    std::cout << "║  From Crash:        " 
              << std::setw(6) << transition_matrix[1][0] << "      " 
              << std::setw(6) << transition_matrix[1][1] << "              ║\n";
}

void RegimeSwitching::printCurrentState() const {
    std::cout << "\n║               Current Regime State                   ║\n";
    
    RegimeParameters params = getCurrentParameters();
    std::cout << "║  Regime: " << std::setw(42) << std::left << params.name << "║\n";
    std::cout << "║  Drift (μ): " << std::fixed << std::setprecision(3) << std::setw(35) << std::right << params.mu << " ║\n";
    std::cout << "║  Volatility (σ): " << std::setw(35) << params.sigma << " ║\n";
}

void RegimeSwitching::printStatistics() const {
    if (total_steps == 0) {
        std::cout << "\nNo statistics available yet (no steps taken).\n";
        return;
    }

    std::cout << "\n║            Regime-Switching Statistics               ║\n";
    std::cout << "║  Total Steps: " << std::setw(38) << total_steps << " ║\n";
    
    // Empirical distribution
    auto [emp_normal, emp_crash] = getEmpiricalDistribution();
    std::cout << "║  Empirical Distribution:                             ║\n";
    std::cout << "║    Normal: " << std::fixed << std::setprecision(3) << std::setw(41) << emp_normal << " ║\n";
    std::cout << "║    Crash:  " << std::setw(41) << emp_crash << " ║\n";
    
    // Theoretical distribution
    auto [theo_normal, theo_crash] = getStationaryDistribution();
    std::cout << "║  Theoretical (Stationary):                           ║\n";
    std::cout << "║    Normal: " << std::setw(41) << theo_normal << " ║\n";
    std::cout << "║    Crash:  " << std::setw(41) << theo_crash << " ║\n";
    
    // Transition counts
    std::cout << "\n║  Transition Counts:                                ║\n";
    std::cout << "║    Normal → Normal: " << std::setw(30) << transition_counts[0][0] << " ║\n";
    std::cout << "║    Normal → Crash:  " << std::setw(30) << transition_counts[0][1] << " ║\n";
    std::cout << "║    Crash → Normal:  " << std::setw(30) << transition_counts[1][0] << " ║\n";
    std::cout << "║    Crash → Crash:   " << std::setw(30) << transition_counts[1][1] << " ║\n";
}

std::pair<double, double> RegimeSwitching::getStationaryDistribution() const {
    /* π = π * P
    [πN​,πC​] = [πN​,πC​] *（Pnn Pnc)
                        (Pcn Pcc) */
    
    // πN ​= πN * ​PNN ​+ πC * ​PCN​
    // πN ​= πN * ​PNC ​+ πC * ​PCC
    // πN + πC ​= 1

    // Solve for the above equation set: for 2-state Markov chain: π = [p_cn / (p_nc + p_cn), p_nc / (p_nc + p_cn)]
    double p_nc = transition_matrix[0][1];  // Normal → Crash
    double p_cn = transition_matrix[1][0];  // Crash → Normal
    
    double sum = p_nc + p_cn;
    if (sum < 1e-10) {  // Avoid division by zero
        return {0.5, 0.5};
    }
    
    double pi_normal = p_cn / sum;
    double pi_crash = p_nc / sum;
    
    return {pi_normal, pi_crash};
}

std::pair<double, double> RegimeSwitching::getExpectedDurations() const {
    // Expected duration = 1 / (prob of leaving state)
    double duration_normal = 1.0 / transition_matrix[0][1];  // 1 / P(Normal→Crash)
    double duration_crash = 1.0 / transition_matrix[1][0];   // 1 / P(Crash→Normal)
    
    return {duration_normal, duration_crash};
}

std::pair<double, double> RegimeSwitching::getEmpiricalDistribution() const {
    if (total_steps == 0) {
        return {0.0, 0.0};
    }
    
    double emp_normal = static_cast<double>(regime_counts[0]) / total_steps;
    double emp_crash = static_cast<double>(regime_counts[1]) / total_steps;
    
    return {emp_normal, emp_crash};
}
