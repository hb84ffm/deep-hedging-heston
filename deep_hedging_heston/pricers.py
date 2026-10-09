# pricers.py
import numpy as np   # only dependency: numpy arrays

def heston_call_deltas(stock_prices, variance, tau, heston_parameters, N=256, chunk=8192):
    """COS pricer for the European call in the Heston model.

    - Evaluates the no-arbitrage price u(t, s, v) and both model deltas of
    Eq. (5.6). With delta^1 = ∂u/∂s (stock) and delta^2 = (∂u/∂v)/(∂L/∂v)
    (variance-swap units, L the variance swap of Eq. 5.4), for a batch
    of states (s, v, tau).

    - Built on the Fourier-cosine series expansion of Fang & Oosterlee (2008)
    driven by the Heston characteristic function. 

    - Truncation [a,b] after the 10-SD rule, N = 256 terms, vectorized and chunked for 10^6 points.

    - Returns: price, delta_stocks, delta_variance_swaps, each of (len(s),).
    """
    kappa = heston_parameters.kappa                                                         # mean-reversion speed of the variance
    theta = heston_parameters.theta                                                         # long-run variance
    xi = heston_parameters.xi                                                               # vol-of-vol
    rho = heston_parameters.rho                                                             # correlation stock/variance
    K = heston_parameters.K                                                                 # strike

    stock_prices = np.atleast_1d(np.asarray(stock_prices, float))                           # price value(s) at ONE point in time (not a path matrix!)
    variance = np.atleast_1d(np.asarray(variance, float))                                   # variance value(s) at the same point in time
    log_stock_prices = np.log(stock_prices)                                                 # log-spot, evaluation argument of the series
    log_strike = np.log(K)                                                                  # log-strike, constant of the payoff coefficients
    log_forward = log_stock_prices                                                          # log-forward = log-spot (r = 0)
    half_width = 10.0 * np.sqrt(variance * tau) + np.abs(log_stock_prices - log_strike)     # half-width: 10 standard deviations plus distance to the strike
    a, b = (log_forward - half_width).min(), (log_forward + half_width).max()               # lower/upper bound of the truncation interval [a,b]  [COS notation]
    interval_width = b - a                                                                  # width of [a,b], normalization of the series
    u = np.arange(N) * np.pi / interval_width                                               # cosine frequencies u_k = k*pi/(b-a)  [COS notation]

    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (u ** 2 + 1j * u))             # sqrt term of the Heston CF  [paper notation]
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)                   # ratio term of the Heston CF  [paper notation]
    D = (kappa - rho * xi * 1j * u - d) / xi ** 2 * (1 - np.exp(-d * tau)) / (1 - g * np.exp(-d * tau))   # CF coefficient in v; its derivative w.r.t. v = δ²
    C = (kappa * theta / xi ** 2) * ((kappa - rho * xi * 1j * u - d) * tau
                                     - 2 * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))   # remaining CF exponent  [paper notation]

    chi = np.real((np.exp(b + 1j * u * (b - a)) - np.exp(log_strike + 1j * u * (log_strike - a))) / (1 + 1j * u))   # payoff coefficient 1  [COS notation]
    psi = np.empty(N)                                                                       # payoff coefficient 2  [COS notation]
    psi[0] = b - log_strike                                                                 # psi at u=0: limit value (avoids division by 0)
    psi[1:] = np.real((np.exp(1j * u[1:] * (b - a)) - np.exp(1j * u[1:] * (log_strike - a))) / (1j * u[1:]))   # psi for k>=1
    wk = np.ones(N); wk[0] = 0.5                                                            # series weights (first term half-weighted)  [COS notation]
    cos_coefficients = wk * (chi - np.exp(log_strike) * psi) * np.exp(C + 1j * u *(-a))     # final series coefficients: state-INdependent, computed once before the loop
    series_norm = 2.0 / interval_width                                                      # normalization factor of the COS series
    dL_v = (1 - np.exp(-kappa * tau)) / kappa                                               # ∂_v L sensitivity of the variance swap L(t,v) from Eq. 5.6: -> variance-swap units  [paper notation]

    price = np.empty(stock_prices.size)                                                     # output: price
    delta_stocks = np.empty(stock_prices.size)                                              # output: δ¹ = number of shares
    delta_variance_swaps = np.empty(stock_prices.size)                                      # output: δ² = number of variance-swap units

    for i in range(0, stock_prices.size, chunk):                                            # blocks of `chunk` points (saves memory for 10^6 paths)
        block = slice(i, i + chunk)                                                         # current block
        state_factor = np.exp(1j * np.outer(log_stock_prices[block], u) +
                              np.outer(variance[block], D))                                 # state-dependent part: exp(i·u·log spot + D·variance)
        series_terms = state_factor * cos_coefficients                                      # state times coefficients = series terms
        price[block] = series_norm * np.real(series_terms).sum(1)                           # price: sum over all frequencies
        delta_stocks[block] = series_norm * np.real(1j * u * series_terms).sum(1) / stock_prices[block]   # δ¹ = ∂/∂log spot (factor i·u), chain rule: /spot
        delta_variance_swaps[block] = series_norm * np.real(D * series_terms).sum(1) / dL_v # δ² = ∂/∂variance (factor D), /∂_v L -> variance-swap units
    return price, delta_stocks, delta_variance_swaps

def model_deltas_on_paths(market_paths, heston_parameters):
    """Model-delta hedge on simulated paths: the benchmark of paper DeepHedging/Bühler , Sec. 5.2].

    - Wrapper that applies heston_call_deltas() (see above!) for every rebalancing date k.
    - It evaluates Eq. (5.6) on all paths with remaining maturity tau = T - k*dt, using
    the column of the path matrices at date k as the state. 
    - This yields the 'model hedge' that the deep hedge is compared against.
    - Returns: delta_stocks, delta_variance_swaps for each (n_paths, timesteps).
    """
    n_paths = market_paths.Stocks.shape[0]                                                  # number of paths read off the matrix
    timesteps = heston_parameters.timesteps                                                 # rebalancing dates
    dt = heston_parameters.dt                                                               # step size
    T = heston_parameters.T                                                                 # total maturity

    delta_stocks = np.empty((n_paths, timesteps))                                           # output: δ¹ per path and date
    delta_variance_swaps = np.empty((n_paths, timesteps))                                   # output: δ² per path and date

    for k in range(timesteps):                                                              # one pricer call per date (column slice = all paths)
        _, delta_s, delta_v = heston_call_deltas(market_paths.Stocks[:, k], market_paths.Variance[:, k], T - k*dt, heston_parameters)
        delta_stocks[:, k] = delta_s                                                        # write results into the columns
        delta_variance_swaps[:, k] = delta_v
    return delta_stocks, delta_variance_swaps