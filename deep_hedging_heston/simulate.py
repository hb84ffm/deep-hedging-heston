# simulate.py
import numpy as np   

class MarketPaths:
    """Container for one batch of simulated market paths."""
    def __init__(self, Stocks, Variance, Variance_swap, Payoff):
        self.Stocks = Stocks                  # (n paths, timesteps+1)
        self.Variance = Variance              # (n paths, timesteps+1)
        self.Variance_swap = Variance_swap    # (n paths, timesteps+1)
        self.Payoff = Payoff                  # (n paths,)

def simulate_market(heston_parameters, seed, nr_of_simulated_paths):
    """
    - Generates Heston paths.
    - Exact CIR sampling for the variance.
    - Simplified Broadie-Kaya sampling for the stock [LBAK04, Sec. 4.2.2]. 
    - Fixed order of random draws: the same seed reproduces bit-identical paths.
    - Returns a MarketPaths object.
    """

    rng = np.random.default_rng(seed)           # own random number generator per run
    dt = heston_parameters.dt
    timesteps = heston_parameters.timesteps

    # CIR constants 
    cir_degrees_of_freedom = 4*heston_parameters.kappa*heston_parameters.theta/heston_parameters.xi**2
    cir_scaling = heston_parameters.xi**2 * (1 - np.exp(-heston_parameters.kappa*dt)) / (4*heston_parameters.kappa)

    # Variance: exact sampling from the CIR transition density
    variance = np.empty((nr_of_simulated_paths, timesteps+1))
    variance[:, 0] = heston_parameters.v0
    for t in range(timesteps):
        lambda_value = 4*heston_parameters.kappa*np.exp(-heston_parameters.kappa*dt)*variance[:, t] / (heston_parameters.xi**2*(1-np.exp(-heston_parameters.kappa*dt)))
        variance[:, t+1] = cir_scaling * rng.noncentral_chisquare(cir_degrees_of_freedom, lambda_value)

    # Integrated variance (trapezoidal rule) — used for the variance swap
    I_t = dt/2 * (variance[:, :-1] + variance[:, 1:])
    variance_integrated = np.zeros((nr_of_simulated_paths, timesteps+1))
    variance_integrated[:, 1:] = np.cumsum(I_t, axis=1)

    # Stocks via the Broadie-Kaya identity (second block of random draws)
    Zt = rng.standard_normal((nr_of_simulated_paths, timesteps))
    log_stocks = np.empty((nr_of_simulated_paths, timesteps+1))
    log_stocks[:, 0] = np.log(heston_parameters.S0)
    for t in range(timesteps):
        log_stocks[:, t+1] = (log_stocks[:, t]  + heston_parameters.rho*
                              (variance[:, t+1] - variance[:, t] - heston_parameters.kappa*heston_parameters.theta*dt + heston_parameters.kappa*I_t[:, t]) / 
                              heston_parameters.xi - 0.5*I_t[:, t] + np.sqrt((1 - heston_parameters.rho**2)*I_t[:, t])*Zt[:, t])
    stocks = np.exp(log_stocks)

    # Variance swap, Eq. (5.4)
    tau = heston_parameters.T - dt*np.arange(timesteps+1)
    variance_swap = variance_integrated + ((variance - heston_parameters.theta)/heston_parameters.kappa)*(1-np.exp(-heston_parameters.kappa*tau)) + heston_parameters.theta*tau

    # Payoff European Call
    payoff = np.maximum(stocks[:, -1] - heston_parameters.K, 0.0)

    return MarketPaths(Stocks=stocks, Variance=variance, Variance_swap=variance_swap, Payoff=payoff)