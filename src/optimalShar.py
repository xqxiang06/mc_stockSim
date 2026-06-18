import numpy as np
import scipy.optimize as sco

def optimizer(mu, cov, rf, mode='max_sharpe'):
    """
    Find portfolio weights that 1)maximize the Sharpe ratio.
                                2)minimize volatility
    
    Parameters
        mu  : list of 3 floats  — annualized mean returns
        cov : 3x3 list of lists — annualized covariance matrix
        rf  : float             — risk-free rate
    
    Returns
        dict with keys: weights, ret, vol, sharpe
    """
    mu = np.array(mu)
    cov = np.array(cov)

    def port_ret(weights):
        return np.dot(weights, mu)
    
    def port_vol(weights):
        # sqrt(w @ cov @ w)
        return np.sqrt(
            np.dot(weights.transpose(),
                   np.dot(cov, weights))
        )
    
    def neg_sharpe(weights):
        return -( (port_ret(weights)) - rf ) / port_vol(weights)
    
    # constraints and bounds
    noa = len(mu)
    # # weights sum to 1, no shorting
    cons = ({'type': 'eq',
             'fun': lambda w: np.sum(w) - 1})
    
    bnds = tuple((0,1) for w in range(noa))
    '''
    # if constrained to force diversification
    bnds = ((0.10, 0.70),   # VOO:  min 10%, max 70%
            (0.10, 0.50),   # VXUS: min 10%, max 50%
            (0.10, 0.40))   # BND:  min 10%, max 40%
    '''
    # start guessing: Equal weights vector
    eweights = np.array(noa * [1.0 / noa,])

    # switch the mode: max_sharpe OR min_vol
    objective = neg_sharpe if mode == 'max_sharpe' else port_vol

    opts = sco.minimize(objective, eweights,
                        method='SLSQP', # Sequential Least Squares Programming
                        bounds=bnds,
                        constraints=cons)
    # the optimal weights vector
    w = opts.x

    return {
        'weights': w.tolist(),   # [w_voo, w_vxus, w_bnd]
        'ret':     float(port_ret(w)),
        'vol':     float(port_vol(w)),
        'sharpe':  float(-opts.fun)
                        # the final function value--the negative Sharpe
    }
