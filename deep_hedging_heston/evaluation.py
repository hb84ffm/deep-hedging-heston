# evaluation.py
import numpy as np                                             # metrics on finished numbers
import tensorflow as tf                                        # forward pass of the model
from .pricers import model_deltas_on_paths                     # benchmark deltas from the package


class HedgeEvaluation:
    """Evaluates a FINISHED, calibrated model on a dataset: 
    - epsilon statistics,J*, sorted CVaR, w vs VaR, p0 = q + J*, and the model-hedge benchmark.
    - Read-only: changes nothing in the model, trains nothing. 
    - Must run AFTER BatchNorm calibration and measures with training=False (calibrated moving
    statistics)
    - the stale-statistics artefact of the old setup is
    structurally excluded here."""

    def __init__(self, deep_hedging_model, heston_parameters):
        self.model = deep_hedging_model                        # finished model (read-only)
        self.alpha = deep_hedging_model.alpha                  # CVaR level from the model
        self.premium = float(deep_hedging_model.premium)       # q for p0 = q + J*

    def terminal_hedge_errors(self, data_tf):
        """Terminal hedging error per path, epsilon = q - Z + (δ·S)_T, evaluated with
        training=False. Returns a numpy array of length n_paths."""
        return self.model.compute_terminal_hedge_error(data_tf, training=False).numpy()

    def sorted_cvar(self, data_tf):
        """Sorted CVaR_alpha of the losses: mean of the worst (1 - alpha) fraction.
        Independent control definition of Eq. (4.6) — not differentiable, pure metric."""
        losses = -self.terminal_hedge_errors(data_tf)
        tail = int((1.0 - self.alpha) * len(losses))
        return float(np.sort(losses)[::-1][:tail].mean())

    def evaluate(self, data_tf, name=""):
        """Full report on one dataset: 
        - mean/std of epsilon, J* = OCE loss (Eq. 4.6) with final w on full loss vector, sorted CVaR, w vs VaR_alpha
        (quantile check), and the indifference price p0 = q + J* (Prop. 3.10(ii)).
        - Returns a dict incl. hedge_errors for the plots."""
        hedge_errors = self.terminal_hedge_errors(data_tf)
        losses = -hedge_errors
        objective = float(self.model.compute_oce_loss(tf.constant(losses, dtype=self.model.premium.dtype)))
        quantile = float(np.quantile(losses, self.alpha))
        result = dict(
            name=name,
            mean_error=float(hedge_errors.mean()),
            std_error=float(hedge_errors.std()),
            J_star=objective,
            sorted_cvar=self.sorted_cvar(data_tf),
            oce_threshold=float(self.model.oce_threshold),
            var_quantile=quantile,
            p0=self.premium + objective,
            hedge_errors=hedge_errors)
        print(f"[{name}] mean = {result['mean_error']:+.4f} | std = {result['std_error']:.4f} | "
              f"J* = {result['J_star']:.4f} | sorted CVaR = {result['sorted_cvar']:.4f} | "
              f"w = {result['oce_threshold']:.4f} vs VaR = {result['var_quantile']:.4f} | "
              f"p0 = {result['p0']:.4f}")
        return result

    def model_hedge_errors(self, market_paths, heston_parameters):
        """Benchmark: 
        - epsilon = q - Z + Σ_k δ^model_k · ΔS_k with the exact model
        deltas of Eq. (5.6) (COS pricer). 
        - Network-free, therefore bit-identical across runs, the strongest consistency check of the pipeline."""
        delta_stocks, delta_variance_swaps = model_deltas_on_paths(market_paths, heston_parameters)
        stocks_increment = np.diff(market_paths.Stocks, axis=1)
        variance_increment = np.diff(market_paths.Variance_swap, axis=1)
        return (self.premium - market_paths.Payoff
                + (delta_stocks * stocks_increment + delta_variance_swaps * variance_increment).sum(axis=1))

    def compare_with_model_hedge(self, data_tf, market_paths, heston_parameters, name=""):
        """Figure-2-style comparison on the same paths: deep hedge vs model hedge
        (mean, std, sorted CVaR). Returns (deep_result, model_result) dicts."""
        deep_result = self.evaluate(data_tf, name=name + " Deep Hedge")
        model_errors = self.model_hedge_errors(market_paths, heston_parameters)
        model_cvar = float(np.sort(-model_errors)[::-1][:int((1.0 - self.alpha) * len(model_errors))].mean())
        print(f"[{name} Model Hedge] mean = {model_errors.mean():+.4f} | std = {model_errors.std():.4f} | "
              f"CVaR_{self.alpha} = {model_cvar:.4f}")
        return deep_result, dict(mean_error=float(model_errors.mean()), std_error=float(model_errors.std()),
                                 sorted_cvar=model_cvar, hedge_errors=model_errors)