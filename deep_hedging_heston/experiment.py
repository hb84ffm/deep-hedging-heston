# experiment.py
import numpy as np                     # arrays, cumulation
import matplotlib.pyplot as plt        # the 4 chart rows
from .simulate import simulate_market  # package wiring instead of notebook globals
from .pricers import model_deltas_on_paths
from .modeldesign import to_tf


def compute_pnl_path(deltas, 
                     stocks_increment, 
                     variance_increment):
    """Paper PnL as a running series: 
    - periodic contributions δ_k·ΔS_k, cumulated up to each date k.
    - deltas: (n_paths, timesteps, 2) | increments: (n_paths, timesteps)
    - Returns: pnl (n_paths, timesteps+1), pnl[:,0] = 0, pnl[:,k] = Σ_{j<k} δ_j·ΔS_j.
    - ONE building block for both hedges: works with formula deltas AND ML deltas (DRY)."""
    periodic = deltas[..., 0] * stocks_increment + deltas[..., 1] * variance_increment   # contribution per period
    pnl = np.zeros((deltas.shape[0], deltas.shape[1] + 1))                               # n+1 dates
    pnl[:, 1:] = np.cumsum(periodic, axis=1)                                             # cumulated
    return pnl

def plot_experiment(time_all, 
                    time_delta, 
                    stocks, variance,
                    deltas_formula, 
                    deltas_ml, 
                    pnl_formula, 
                    pnl_ml,
                    portfolio_formula, 
                    portfolio_ml, 
                    payoff, 
                    premium,
                    eps_formula, 
                    eps_ml, 
                    seed, 
                    history=None):
    """4 rows of 2 charts each. Pure plotting: 
    - takes numpy numbers only from run_experiment or any other source.
    - Shape contracts: stocks/variance/pnl/portfolio = (n+1,), deltas = (n, 2).
    - history: [(step, J_train, J_val), ...] from training, None/empty = placeholder text."""
    BLUE = 'tab:blue'
    ORANGE = 'tab:orange'
    L_ANALYTICAL = "Analytical"
    L_ML = "ML model"

    fig, axes = plt.subplots(4, 2, figsize=(14, 11))

    # Row 1: market
    axes[0, 0].plot(time_all, stocks)
    axes[0, 0].set_title("Stock price")
    axes[0, 1].plot(time_all, variance)
    axes[0, 1].set_title("Variance")

    # Row 2: deltas — both strategies in the same chart
    axes[1, 0].plot(time_delta, deltas_formula[:, 0], color=BLUE,   label=L_ANALYTICAL)
    axes[1, 0].plot(time_delta, deltas_ml[:, 0],      color=ORANGE, label=L_ML)
    axes[1, 0].set_title("δ¹ Stock — Analytical vs. ML model"); axes[1, 0].legend()

    axes[1, 1].plot(time_delta, deltas_formula[:, 1], color=BLUE,   label=L_ANALYTICAL)
    axes[1, 1].plot(time_delta, deltas_ml[:, 1],      color=ORANGE, label=L_ML)
    axes[1, 1].set_title("δ² VarSwap — Analytical vs. ML model"); axes[1, 1].legend()

    # Row 3: PnL & portfolio — reference lines: payoff (grey) and premium q (black), solid
    axes[2, 0].plot(time_all, pnl_formula, color=BLUE,   label=L_ANALYTICAL)
    axes[2, 0].plot(time_all, pnl_ml,      color=ORANGE, label=L_ML)
    axes[2, 0].set_title("PnL — Analytical vs. ML model"); axes[2, 0].legend()

    axes[2, 1].plot(time_all, portfolio_formula, color=BLUE,   label=L_ANALYTICAL)
    axes[2, 1].plot(time_all, portfolio_ml,      color=ORANGE, label=L_ML)
    axes[2, 1].axhline(payoff,  lw=1, color='grey',  label=f"Payoff = {payoff:.2f}")
    axes[2, 1].axhline(premium, lw=1, color='black', label=f"Premium q = {premium:.2f}")
    axes[2, 1].set_title("Portfolio — Analytical vs. ML model"); axes[2, 1].legend()

    # Row 4 left: terminal error — horizontal bars, x-axis symmetric around 0
    bar_labels, bar_values = [L_ANALYTICAL, L_ML], [eps_formula, eps_ml]
    axes[3, 0].barh(bar_labels, bar_values, color=[BLUE, ORANGE])
    axes[3, 0].axvline(0, color='grey', lw=0.8)
    m = max(abs(eps_formula), abs(eps_ml))
    if m == 0: m = 1.0                                                                  # fallback: both values exactly 0 (unlikely)
    axes[3, 0].set_xlim(-1.35 * m, 1.35 * m)                                            # symmetric, 35% margin for the number annotations
    for i, value in enumerate(bar_values):
        offset = 0.02 * m * (1 if value >= 0 else -1)
        axes[3, 0].text(value + offset, i, f"{value:+.4f}", va='center',
                    ha='left' if value >= 0 else 'right', fontweight='bold')
    axes[3, 0].set_title("Terminal error:  epsilon = q − Payoff + Σ δ·ΔS", fontsize=10)

    # Row 4 right: learning curve — OCE loss J over training steps (from train_history)
    if history:                                                                         # present after run() or save_model + load_model
        steps   = [point[0] for point in history]
        j_train = [point[1] for point in history]
        j_val   = [point[2] for point in history]
        axes[3, 1].plot(steps, j_train, color='darkgoldenrod', label="Training")
        axes[3, 1].plot(steps, j_val,   color='black', label="Validation")
        axes[3, 1].set_xlabel("Steps")
        axes[3, 1].legend(fontsize=8)
    else:                                                                               # old models without a saved curve
        axes[3, 1].text(0.5, 0.5, "no training history available\n(old model without saved curve)",
                   ha='center', va='center', transform=axes[3, 1].transAxes, color='grey')
    axes[3, 1].set_title("Learning curve: OCE loss J", fontsize=10)

    for row in range(3):                                                                # x-labels only for the time series
        axes[row, 0].set_xlabel("Date k"); axes[row, 1].set_xlabel("Date k")

    fig.suptitle(f"Deep hedging experiment — seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def run_experiment(model, heston_parameters, seed):
    """ONE random path (only the seed selectable, paths=1): 
    - runs both hedges on it and assembles the figure via plot_experiment. 
    - Computation ONLY, layout lives in plot_experiment."""
    paths = simulate_market(heston_parameters, seed=seed, nr_of_simulated_paths=1)
    timesteps = heston_parameters.timesteps
    time_all   = np.arange(timesteps + 1)
    time_delta = np.arange(timesteps)
    premium = float(model.premium)

    delta_s, delta_v = model_deltas_on_paths(paths, heston_parameters)
    deltas_formula = np.stack([delta_s, delta_v], axis=-1)

    data_tf = to_tf(paths)
    deltas_ml = model.compute_deltas(data_tf['log_stocks'], data_tf['variance'], training=False).numpy()

    stocks_incr  = np.diff(paths.Stocks, axis=1)
    varswap_incr = np.diff(paths.Variance_swap, axis=1)
    pnl_formula = compute_pnl_path(deltas_formula, stocks_incr, varswap_incr)
    pnl_ml      = compute_pnl_path(deltas_ml, stocks_incr, varswap_incr)
    portfolio_formula = premium + pnl_formula
    portfolio_ml      = premium + pnl_ml

    eps_formula = float(portfolio_formula[0, -1] - paths.Payoff[0])
    eps_ml      = float(portfolio_ml[0, -1] - paths.Payoff[0])

    fig = plot_experiment(time_all, 
                          time_delta, 
                          paths.Stocks[0], 
                          paths.Variance[0],
                          deltas_formula[0], 
                          deltas_ml[0],
                          pnl_formula[0], 
                          pnl_ml[0],
                          portfolio_formula[0], 
                          portfolio_ml[0],
                          float(paths.Payoff[0]), 
                          premium,
                          eps_formula, 
                          eps_ml, 
                          seed,
                          history=getattr(model, 'train_history', None))
    plt.show()
    print(f"Terminal error — formula: {eps_formula:+.4f} | ML model: {eps_ml:+.4f}")
    return dict(figure=fig, eps_formula=eps_formula, eps_ml=eps_ml)