# parameters.py
# Goal: all fixed numbers of the experiment 'Heston market with r=0 and no costs' as in Deep Hedging/Bühler paper in ONE place. 

class HestonParameters:
    """Market parameters: Heston under Q, European call. Defaults follow the paper (Sec. 5.2)."""

    def __init__(self,
                 v0=0.04,               # initial variance
                 S0=100,                # initial spot
                 K=100,                 # strike
                 T=30/365,              # maturity in years
                 timesteps=30,          # number of steps / rebalancing dates
                 rho=-0.7,              # correlation of the Brownian motions
                 kappa=1.0,             # mean-reversion speed of the variance
                 theta=0.04,            # long-run variance
                 xi=2.0):               # vol-of-vol

        self.v0 = v0
        self.S0 = S0
        self.K = K
        self.T = T
        self.timesteps = timesteps
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.dt = T / timesteps


class TrainingParameters:
    """Training and pipeline parameters. Defaults follow the paper (Sec. 5.1), with fixed seeds."""

    def __init__(self,
                 ALPHA=0.5,             # CVaR level (risk aversion)
                 LEARNING_RATE=0.005,   # for Adam optimizer
                 BATCH=256,             # minibatch size
                 N_STEPS=200_000,       # gradient steps
                 PRINT_EVERY=5_000,     # training progress is logged at every PRINT_EVERY gradient step
                 paths=100_000,         # train/val paths
                 paths_test=1_000_000,  # OOS test paths
                 seed_train=15,         # seed for the RNG to generate the training paths
                 seed_val=25,           # seed for the RNG to generate the validation paths
                 seed_test=35,          # seed for the RNG to generate the OOS test paths
                 seed_tf=42):           # TF weight initialization

        self.ALPHA = ALPHA
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH = BATCH
        self.N_STEPS = N_STEPS
        self.PRINT_EVERY = PRINT_EVERY
        self.paths = paths
        self.paths_test = paths_test
        self.seed_train = seed_train
        self.seed_val = seed_val
        self.seed_test = seed_test
        self.seed_tf = seed_tf


class DFNNParameters:
    """Architecture of the DFNNs, one network per rebalancing date.
    - Input: (log S_k, V_k). 
    - Two hidden layers of width d + 15 (paper Sec. 5.1).
    - Output: delta^1 (stock) and delta^2 (variance swap units).
    """

    def __init__(self,
                 nr_of_input_features=2,    # input per network: (log S_k, V_k) with S_k=stock price, V_k=variance
                 nr_of_units=17,            # hidden layer width = d + 15= 2 + 15 = 17 (paper Sec. 5.1)
                 nr_of_outputs=2,           # output: delta^1 (stock) and delta^2 (variance swap)
                 activation='relu',         # activation function
                 oce_threshold_start=0.0):  # initial value for the OCE threshold w

        self.nr_of_input_features = nr_of_input_features
        self.nr_of_units = nr_of_units
        self.nr_of_outputs = nr_of_outputs
        self.activation = activation
        self.oce_threshold_start = oce_threshold_start