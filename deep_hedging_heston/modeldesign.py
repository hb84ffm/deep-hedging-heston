# modeldesign.py
# The complete ML model.
import numpy as np                                                                           # data preparation before computational graph (CG)
import tensorflow as tf                                                                      # networks, variables, CG
from tensorflow import keras                                                                 # Sequential API
from tensorflow.keras import layers                                                          # Dense, BatchNormalization, Activation

DTYPE = tf.float32                                                                           # set numerical precision for all ML computation

def reset_tensorflow(seed_tf):
    """Reset the TF session and make the initial weights reproducible.
    - Clears old networks/graphs and fixes the random seed used for weight initialization. 
    - MUST be called before creating a DeepHedgingModel,otherwise training is not reproducible."""
    tf.keras.backend.clear_session()                                                         # dispose of old networks/graphs
    tf.keras.utils.set_random_seed(seed_tf)                                                  # fixes the random initial weights

# ============================================================================== START CG ==============================================================================
def to_tf(market_paths):
    """Convert numpy paths into TF tensors: the 5 quantities the model's
    computational graph consumes [data preparation, 'before the DFNNs'].

    - Keys of the returned dict are the CG variables:
    log_stocks, variance (features), stocks_increment / variance_increment
    (hedging-instrument increments ΔS¹, ΔS²), payoff (terminal liability Z)."""
    return dict(                                                                             # dict = TF data package; keys = CG variables
        log_stocks=tf.constant(np.log(market_paths.Stocks), DTYPE),                          # feature 1: log S_k
        variance=tf.constant(market_paths.Variance, DTYPE),                                  # feature 2: V_k
        stocks_increment=tf.constant(np.diff(market_paths.Stocks, axis=1), DTYPE),           # ΔS_k (hedging instrument 1, which is stock)
        variance_increment=tf.constant(np.diff(market_paths.Variance_swap, axis=1), DTYPE),  # ΔVS_k (hedging instrument 2, which is variance swap)
        payoff=tf.constant(market_paths.Payoff, DTYPE))                                      # Z (terminal liability, which is payoff!)

def make_single_network(dfnn_parameters):
    """Build ONE untrained DFNN for ONE rebalancing date.

    - Architecture per paper Sec. 5.1 (fixed, not configurable):
    2 -> Dense 17 -> BatchNorm -> ReLU -> Dense 17 -> BatchNorm -> ReLU -> Dense 2.
    - Linear output layer: δ¹ (stock) and δ² (variance swap) remain unbounded.
    - BatchNorm applied immediately before the activation."""
    return keras.Sequential([
        keras.Input(shape=(dfnn_parameters.nr_of_input_features,)),                          # explicit Input layer instead of input_shape in the first Dense
        layers.Dense(dfnn_parameters.nr_of_units),                                           # linear: 2 -> 17
        layers.BatchNormalization(),                                                         # BN before activation
        layers.Activation(dfnn_parameters.activation),                                       # ReLU
        layers.Dense(dfnn_parameters.nr_of_units),                                           # linear: 17 -> 17
        layers.BatchNormalization(),                                                         # BN before activation
        layers.Activation(dfnn_parameters.activation),                                       # ReLU
        layers.Dense(dfnn_parameters.nr_of_outputs),                                         # linear: 17 -> 2
    ])

class DeepHedgingModel:
    """The COMPLETE ML model: one DFNN per rebalancing date + trainable OCE threshold w 
    + the full computational graph
    (features -> deltas -> PnL -> epsilon (= q - Z + PnL) -> loss).

    Loss: OCE representation of CVaR, Eq. (4.6) of [Buh01].
    The gradient step itself (GradientTape, optimizer) deliberately lives
    in modeltraining.py: this class is the graph, that one is the execution."""

    def __init__(self, heston_parameters, dfnn_parameters, premium, training_parameters):
        self.networks = []                                                                   # ONE DFNN per rebalancing date
        for k in range(heston_parameters.timesteps):
            self.networks.append(make_single_network(dfnn_parameters))                       # network k <-> date t_k
        self.oce_threshold = tf.Variable(dfnn_parameters.oce_threshold_start, dtype=DTYPE,   # w of the OCE loss, trained along with the networks
                                         trainable=True, name="oce_threshold")
        self.premium = tf.constant(premium, dtype=DTYPE)                                     # q — belongs to the model: epsilon = q - Z + PnL
        self.alpha = training_parameters.ALPHA                                               # CVaR level of the OCE loss

    def compute_deltas(self, log_stocks, variance, training):
        """[before/in the DFNNs] For every date k: build features (log S_k, V_k)
        and feed them into network k. Returns (batch, timesteps, 2).

        - training=True:  BN uses batch statistics and updates the moving
                        statistics (training / calibration regime)
        - training=False: BN uses the calibrated moving statistics (evaluation)
        - NO default for `training`: passing the wrong flag was bug #1 (J* = 1.19 instead of 0.28)."""
        deltas_per_step = []                                                                 # deltas per date
        for k in range(len(self.networks)):
            features = tf.stack([log_stocks[:, k], variance[:, k]], axis=1)                  # (batch, 2)
            deltas_k = self.networks[k](features, training=training)                         # (batch, 2)
            deltas_per_step.append(deltas_k)
        return tf.stack(deltas_per_step, axis=1)                                             # (batch, timesteps, 2)

    def compute_pnl(self, deltas, stocks_increment, variance_increment):
        """[after the DFNNs] Self-financed PnL of the hedge:
        (δ·S)_T = Σ_k δ_k · ΔS_k, both instruments (paper Sec. 2)."""
        return tf.reduce_sum(deltas[..., 0] * stocks_increment                               # δ¹ (stock) · ΔS
                             + deltas[..., 1] * variance_increment, axis=1)                  # δ² (variance swap) · ΔVS

    def compute_terminal_hedge_error(self, data, training):
        """[complete forward pass] Terminal hedging error per path:
        epsilon = q - Z + (δ·S)_T. Returns (batch,)."""
        deltas = self.compute_deltas(data['log_stocks'], data['variance'], training)
        pnl = self.compute_pnl(deltas, data['stocks_increment'], data['variance_increment'])
        return self.premium - data['payoff'] + pnl                                           # epsilon (= q - Z + PnL) per path: (batch,)

    def compute_losses(self, data, training):
        """Terminal losses L = -epsilon (the OCE/CVaR loss expects losses)."""
        return -self.compute_terminal_hedge_error(data, training)

    def compute_oce_loss(self, losses):
        """OCE/CVaR loss, Eq. (4.6): J = w + E[max(L - w, 0)] / (1 - alpha).

        w = self.oce_threshold (trained together with the networks),
        alpha = CVaR level. Minimizing J over w and the networks jointly
        optimizes the CVaR at level alpha."""
        return self.oce_threshold + tf.reduce_mean(tf.nn.relu(losses - self.oce_threshold)) / (1.0 - self.alpha)
# ============================================================================== END CG ==============================================================================

    def collect_trainable_variables(self):
        """All trainable variables for the optimizer: w first, then the DFNNs
        (per network: 2x(Dense kernel + bias) + 2x(BN gamma + beta) = 10)."""
        variables = [self.oce_threshold]
        for network in self.networks:
            variables.extend(network.trainable_variables)
        return variables