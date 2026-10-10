# Deep Hedging in a Heston Market

This repository implements the Deep Hedging algorithm (see [Buh01]), which hedges a short position in a European call through reinforcement learning.

The synthetic market environment follows Heston's stochastic volatility model, where at each rebalancing step the hedge quantities of stock and idealized variance swap are estimated by deep feed forward neural networks (DFNNs), such that an Optimized Certainty Equivalent (OCE) of a CVaR loss function is minimized.

In contrast to classical delta hedging (optimal only in frictionless, complete markets), the Deep Hedging algorithm directly minimizes the terminal hedging error, without further model assumptions (equivalent Martingale measure, Greek estimation, etc.). The method's computational performance is invariant in (portfolio) size, as it depends mainly on the number of hedging instruments.

To showcase our implementation's correctness, we simulate a synthetic Heston market, for which we show that the ML estimates are close to analytical estimations.

The repository provides an end-to-end pipeline across market simulation, pricing, strategy training, validation, PnL/portfolio calculation and benchmarking of ML estimates against analytical solutions. Furthermore, a trained model `heston_30d_alpha_50` (reproducing the Heston market setting in [Buh01]), is shipped by which the user can experiment with or even train new ML models to run experiments across new market regimes.

## 1. Features

- **Market simulation under Heston** with:
  - exact CIR sampling (noncentral $\chi^2$) for the variance paths $V_t$ (see [LBAK04]) 
  - simplified Broadie-Kaya scheme for stock paths $S_t^1$, based on sampled variance (see [BroKay03]) 
- **Analytical pricing** of:
  - risk-neutral premium $q$ (see [FangOosterlee08])
  - deltas $\delta_t^1, \delta_t^2$ (stock and variance-swap units, see [FangOosterlee08])
- **Machine learning design (computational graph, training, validation)** including:
  - Hedge calculations in execution order:<br> 
     - inputs $(\log S_k^1, V_k)$, $\Delta S^1_k$, $\Delta S^2_k$, payoff $\max \lbrace {S^1_T - K,0}\rbrace$ $(S^1_T - K)^+$, premium $q$<br> 
     - one DFNN per rebalancing step with layers ($2 \to 17 \to 17 \to 2$) with batch normalization before activations<br>
     - deltas $\delta_k$<br> 
     - self-financed PnL  $\sum_k (\delta_k^1 \cdot \Delta S_k^1+\delta_k^2 \cdot \Delta S_k^2)$<br>
     - portfolio value<br> 
     - terminal hedging error $\varepsilon = q - (S^1_T - K)^+ + \sum_k (\delta_k^1 \cdot \Delta S_k^1+\delta_k^2 \cdot \Delta S_k^2)$<br> 
     - Minimization of loss $J$ as OCE representation of CVaR (due to non-differentiability of CVaR)
  - **Joint training** of all DFNNs and the OCE threshold $w$, with Adam optimizer
  - **Automatic BatchNorm calibration** at the end of training
  - **Evaluation of hedging quality** via $J^\ast$, sorted CVaR, $w$-vs-VaR check and indifference price $p_0 = q + J^\ast$
- **One-command experiments:** to create market scenario, plot charts of market, strategies, PnL, terminal error for ML hedge vs analytical estimations and learning curves
- **Easy design & training of new models** via three parameter blocks (`HestonParameters`, `TrainingParameters`, `DFNNParameters`) with paper defaults as fallback
- **Easy save and load and use** of newly trained models
  
## 2. Requirements
- Python 3.10+ (tested with 3.11.9)
- numpy==2.1.3
- tensorflow==2.19.0
- matplotlib==3.10.5

Exact pinned versions as in `requirements.txt`, required to reproduce the bit-identical reference values.
## 3. Installation

Clone the repository:
```bash
git clone https://github.com/hb84ffm/deep-hedging-heston.git
cd deep-hedging-heston
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

Install the pinned dependencies, then the package itself (editable):
```bash
pip install -r requirements.txt
pip install -e .
```

To run the example Jupyter notebooks:
```bash
pip install notebook
```

## 4. Usage

There are two ways to obtain a model.

### 1. Load and use the shipped model `heston_30d_alpha_50`:

```python
from deep_hedging_heston import DeepHedger
dh = DeepHedger.load_model("ml_models", "heston_30d_alpha_50") # load the model, make sure you set the correct path to where the model is stored
dh.experiment(seed=4) # runs the experiment, where seed=4 is the reference experiment from [Buh01] with terminal errors -1.0672 (analytical) / -1.0272 (ML)
```

### 2. Train your own model:

```python
from deep_hedging_heston import DeepHedger
from deep_hedging_heston.parameters import HestonParameters, TrainingParameters, DFNNParameters  # import parameter blocks to build custom model 
dh_custom = DeepHedger( # specify the desired parameters
    heston_parameters=HestonParameters(
        v0=0.05,                
        S0=120,                 
        K=105,                  
        T=15/365,               
        timesteps=15,           
        rho=-0.5,               
        kappa=1.5,              
        theta=0.05,             
        xi=1.8),

    training_parameters=TrainingParameters(
        ALPHA=0.9,              
        LEARNING_RATE=0.005,    
        BATCH=256,              
        N_STEPS=200_000,                
        PRINT_EVERY=10_000,     
        paths=50_000,           
        paths_test=200_000,     
        seed_train=101,         
        seed_val=202,           
        seed_test=303,          
        seed_tf=7),

    dfnn_parameters=DFNNParameters(
        nr_of_units=17,         
        activation='tanh',      
        oce_threshold_start=0.0))
train_model = dh_custom.run() # train the specified ml model
dh_custom.save_model("ml_models", "heston_15d_alpha_90") # save the model to folder with name, folder is "ml_models" and name is "heston_15d_alpha_90"  
my_custom_model = DeepHedger.load_model("ml_models", "heston_15d_alpha_90")  # load the saved model  
my_custom_model.experiment(seed=75) # run an experiment
```

**Interface:**
| Method | Arguments | What it does | Returns |
|---|---|---|---|
| `DeepHedger(...)` | 3 optional parameter blocks | builds the model and computes the premium $q$ (seconds, no training) | `DeepHedger` object |
| `.run()` | none (all settings come from constructor) | simulates, trains, calibrates, evaluates | results dict with premium $q$, metrics on validation, out-of-sample data and comparison of the ML hedge against the analytical benchmark. |
| `.save_model(folderpath, name)` | folder, name | writes the checkpoint and the parameter JSON including the learning curve | - |
| `DeepHedger.load_model(folderpath, name)` | folder, name | rebuilds the model from the files, no training | `DeepHedger` object |
| `.experiment(seed)` | non-negative integer | one scenario, both hedges, 4 × 2 chart | dict with figure and both terminal errors |
| `HestonParameters(...)` | 9 optional parameters | instantiates the object | `HestonParameters` object |
| `TrainingParameters(...)` | 11 optional parameters | instantiates the object | `TrainingParameters` object |
| `DFNNParameters(...)` | 5 optional parameters | instantiates the object | `DFNNParameters` object |

**We provide detailed example workflows on how to load, train, save and run the ML models, please see folder `examples/` under**:
- `01_load_the_basemodel_and_start_experiment.ipynb` loads the shipped model `heston_30d_alpha_50` to run reference experiment from [Buh01] and test it on new market scenarios.
- `02_train_your_own_model_and_start_experiment.ipynb` configures a new ML model with market and risk preferences, trains, saves, loads the ML model plus experiments with it.
## 5. Package structure

```txt
deep-hedging-heston/
├── deep_hedging_heston/      <- the package
│   ├── __init__.py
│   ├── parameters.py         <- parameter definition for market, training and network
│   ├── simulate.py           <- exact CIR + simplified Broadie-Kaya calculation
│   ├── pricers.py            <- Analytical pricer (Fang-Oosterlee) for premium + deltas
│   ├── modeldesign.py        <- computational graph, DFNNs, batch normalization, OCE threshold
│   ├── modeltraining.py      <- Adam loop, learning curve, calibration of batch normalization statistics 
│   ├── evaluation.py         <- J*, sorted CVaR, p0 = q + J*, benchmark
│   ├── experiment.py         <- single-scenario charts and terminal errors
│   └── orchestrate.py        <- the orchestration wrapper
├── examples/                 <- the two example notebooks for users
├── ml_models/                <- shipped model: checkpoint + parameters
├── tests/                    <- pytest reference-value tests
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```
## 6. Reference values

The shipped model `heston_30d_alpha_50` reproduces the Heston-reference experiment of [Buh01], very closely. The numbers below are our exact results the shipped model produces and the test suite (`pytest tests/`) recomputes them from the checkpoint. A passing testrun therefore certifies that your installation reproduces these results, if you modify the pipeline code, a failing test tells you which of these quantities your change has shifted.

| Quantity | This repository | Paper [Buh01] |
|---|---|---|
| Risk-neutral premium $q$ | 1.6918 | 1.69 |
| Validation $p_0 = q + J^\ast$ | 1.9693 | 1.94 |
| Out-of-sample $J^\ast$ / $p_0$ | 0.2747 / 1.9666 | — |
| Model hedge (validation), std / sorted CVaR | 0.3808 / 0.2433 | — |
| Experiment seed 4: terminal error analytical / ML | −1.0672 / −1.0272 | — |

**Reproducibility notes:**
- With the versions of `requirements.txt`, the shipped model reproduces the numbers exactly, but on different platform (other CPU), reproducibility holds in distribution, but not bit-by-bit.
- The parameter JSON stores **what the model is** (weights, $w$, BatchNorm statistics), not **how it was trained**, hence learning rate, batch size, number of steps and data seeds are not saved. A freshly retrained model therefore matches distributional quantities, but not single-path values such as the seed-4 experiment.

## 7. Methodology and implementation

In this section we outline the problem statement, give a short intro to the Heston model, explain the Deep Hedging method and provide our algorithm for market design, training, validation and benchmarking.

### 7.1 Problem statement

Consider a Heston market driven by two stochastic processes, $S_t^1$ (stock) and $V_t$ (variance). At $t=0$ a European call is sold for premium $q$, while at $t=T$ the payoff $Z:=\max\lbrace S_T^1-K,0\rbrace$ must be delivered. Since $S_T^1:\Omega\to\mathbb{R_{+}}$ is a random variable, the liability from $Z$ is uncertain and must be hedged by trading in $S_t^1$ and in an idealized variance swap $S_t^2$ (since $V_t$ is not tradeable!).

Trading occurs at discrete timesteps $t_k=k \cdot \mathrm{d}t$ with $\mathrm{d}t=T/n$ for $k=0,\ldots,n-1$, where at each timestep the positions $\delta_k=(\delta^1_k,\delta^2_k)$ are chosen and held until next rebalancing date, to hedge the liability. The strategy is assumed to be self-financing (no external cash flows after premium received!), which yields the terminal hedging error $\varepsilon$ by 

$$
\varepsilon(\delta):=\underbrace{q}_{\text{premium}}-\underbrace{Z}_{\text{payoff}} +\sum_{k=0}^{n-1} \Bigl( \underbrace{\delta^1_k}_{\text{quantity}} \cdot \underbrace{(S^1_{k+1}-S^1_k)}_{\text{stock change}}+\underbrace{\delta^2_k}_{\text{quantity}} \cdot \underbrace{(S^2_{k+1}-S^2_k)}_{\text{VS change}}\Bigr)
$$

Rather than minimizing $\varepsilon$, risk is measured via the CVaR-loss at level $\alpha \in [0, 1)$. To further optimize the gradient estimation, the OCE representation of CVaR is used, to yield 


$$ 
J(\delta, w)=w + \frac{1}{1-\alpha}\mathbb{E} [\max \lbrace -(q-Z +\sum_{k=0}^{n-1} (\delta^1_k \cdot (S^1_{k+1}-S^1_k)+\delta^2_k \cdot (S^2_{k+1}-S^2_k))) - w, 0\rbrace] 
$$

The optimization problem then results in finding $J^\ast$, the minimal achievable CVaR-based residual hedging risk over all admissible hedging strategies, given by 

$$
J^\ast=\inf_{\delta, w}J(\delta,w)
$$ 

### 7.2 Heston market

The stochastic processes $S_t^1$ and $V_t$ from section 7.1 are specified by Heston's stochastic volatility model under the risk-neutral measure $\mathbb{Q}$. The model defines parameters $r \ge 0$ (rate), $V_0>0$ (initial variance), $\theta_{par}>0$ (long-run mean variance), $\kappa>0$ (mean-reversion speed), $\xi>0$ (volatility of volatility), $\rho\in[-1,1]$ (correlation of stock and variance shocks) and is for $t>0$ given as a system of stochastic differential equations (SDEs) 

$$
\mathrm{d}S_t^1 = \sqrt{V_t} S_t^1 \mathrm{d}W_t^{S_t^1}
$$ 

$$
\mathrm{d}V_t = \kappa(\theta_{par} - V_t)\mathrm{d}t + \xi \sqrt{V_t}\mathrm{d}W_t^{V_t}
$$ 

where $V_t$ follows a Cox-Ingersoll-Ross (CIR) process with initial conditions 

$$
S_0^1>0
$$ 

$$
V_0>0
$$  

Furthermore, both Wiener processes fulfil 

$$
\mathrm{d}\langle W_t^{S_t^1},W_t^{V_t}\rangle =\rho\mathrm{d}t
$$ 

while the expected variance satisfies for $t\to\infty$ 

$$
\mathbb{E}_{\mathbb{Q}}[V_t]=\theta_{par}+(V_0-\theta_{par})e^{-\kappa t}\rightarrow\theta_{par}
$$ 

Since the variance $V_t$ is not tradeable, an idealized variance swap $S_t^2$ is introduced and it's value at $t$ is given by 

$$
S_t^2=\underbrace{\int_0^t V_s\mathrm{d}s}_{\text{realized variance}}+\underbrace{\frac{V_t-\theta_{par}}{\kappa}\left(1-e^{-\kappa(T-t)}\right)+\theta_{par}(T-t)}_{\text{expected future variance}}
$$

Hence price changes $S^1_{k+1}-S^1_k$ and $S^2_{k+1}-S^2_k$ of the hedging error in 7.1 become computable.

### 7.3 Pricing $C_t$ in the Heston market via Fang-Oosterlee

The premium $q$ of 7.1 is the price of the call at $t=0$, under $\mathbb{Q}$ an arbitrary price is given by 

$$
C_t =\mathbb{E}^{\mathbb{Q}}\Big[\underbrace{Z}_{\text{payoff of 7.1}}\Big| S_t^1, V_t\Big]
$$  

Since $(S_t^1, V_t)$ are Markov (future depends only on current state), the price is a function of time, stock and variance given by 

$$
C_t = u(t, S_t^1, V_t)
$$ 

$$
q = u(0, S_0^1, V_0)
$$ 

To compute $u$ and its partial derivatives we use the Fourier-cosine expansion of [FangOosterlee08], evaluated at a generic state $(s, v) = (S_t^1, V_t)$ with remaining maturity $\tau = T - t$.
 
The truncated log-stock interval in our implementation is chosen as (see [Sey15]) 

$$
h = 10\sqrt{v\tau} + \bigl|\ln(s) - \ln( K)\bigr|
$$ 

$$
a = \ln (s - h)
$$ 

$$
b = \ln (s + h)
$$ 

On this interval the price is the cosine series 

$$
u(t, s, v) = \frac{2}{b-a}\sum_{k=0}^{N_{cos}-1}{}' \text{Re}\Big[\underbrace{e^{i u_k \ln(s) + D(u_k)v}}_{\text{state } (s,v)}\underbrace{\bigl(\chi_k - K\psi_k\bigr) e^{C(u_k) - i u_k a}}_{\text{state-independent}}\Big]
$$ 

$$
u_k = \frac{k\pi}{b-a}
$$ 

with $i$ as imaginary unit, $\text{Re}[\cdot]$ as real part, $\sum{}'$ weights the first term of the sum with $1/2$ and $N_{cos}$ is the number of cosine terms. The Heston characteristic function enters through coefficients 

$$
d(u) = \sqrt{(\rho\xi i u - \kappa)^2 + \xi^2(u^2 + i u)}
$$ 

$$
g(u) = \frac{\kappa - \rho\xi i u - d(u)}{\kappa - \rho\xi i u + d(u)}
$$  

$$
C(u) = \frac{\kappa\theta_{par}}{\xi^2}[(\kappa - \rho \xi i u - d(u)) \tau - 2\ln(\frac{1 - g(u) e^{-d(u) \tau}}{1 - g(u)})]
$$ 

$$
D(u) = \frac{\kappa - \rho \xi i u - d(u)}{\xi^2}\cdot\frac{1 - e^{-d(u) \tau}}{1 - g(u) e^{-d(u) \tau}}
$$ 

The payoff coefficients of the call are for $(k \ge 1)$ given by 

$$
\chi_k = \text{Re}[\frac{e^{b + i u_k (b-a)} - e^{\ln(K) + i u_k (\ln(K) - a)}}{1 + i u_k}]
$$ 

$$
\psi_0 = b - \ln(K)
$$ 

$$
\psi_k = \text{Re}\left[\frac{e^{i u_k (b-a)} - e^{i u_k (\ln(K) - a)}}{i u_k}\right]
$$ 

where $\psi_0$ is the limit value at $u_0 = 0$ to avoid zero-division.

Differentiating the series term by term over interval $[a, b]$ (as implemented: one shared interval per batch) gives both analytical deltas (as in Eq. (5.6) of [Buh01]) in closed series form 

$$
\delta_t^1 = \frac{\partial u}{\partial s} = \frac{2}{b-a} \frac{1}{s}\sum_{k=0}^{N_{cos}-1}{}' \text{Re}[i u_k  e^{i u_k \ln(s) + D(u_k)v}(\chi_k - K\psi_k) e^{C(u_k) - i u_k a}]
$$ 

$$
\delta_t^2 = \frac{\partial_v u}{\partial_v L} = \frac{2}{b-a}\frac{1}{\partial_v L(t, v)}\sum_{k=0}^{N_{cos}-1}{}'\text{Re}[D(u_k) e^{i u_k \ln(s) + D(u_k)v} (\chi_k - K\psi_k) e^{C(u_k) - i u_k a}]
$$ 

with $\partial_v L(t,v) = \frac{1}{\kappa}\bigl(1 - e^{-\kappa \tau}\bigr)$ as variance sensitivity of the variance swap from 7.2. 
 
In simple words: $\delta_t^1$ is the sensitivity of the price to the stock, $\delta_t^2$ is the sensitivity to variance divided by the variance sensitivity
of the swap (units of the swap to hold).

In continuous time, the holding $\delta_t = (\delta_t^1, \delta_t^2)$ would replicate the payoff exactly (Eq. (5.5) of [Buh01]) via 

$$
Z = q + \int_0^T \Bigl( \delta_t^1\mathrm{d}S_t^1 + \delta_t^2\mathrm{d}S_t^2 \Bigr)
$$ 

On the discrete grid of 7.1 this replication fails and the residual is the hedging error $\varepsilon(\delta)$. The strategy $\delta_t = (\delta_t^1, \delta_t^2)$ computed from these analytical methods serves therefore as the natural benchmark ([Buh01, Section 5.2]) against which the ML hedge is evaluated.

### 7.4 Path simulation

The training and validation data necessary for Section 7.5 are in our case simulated. Paths of the hedging instruments and the payoff are generated under $\mathbb{Q}$. The variance paths are then drawn from their exact CIR transition density (no Euler discretization) and the stock paths follow the simplified Broadie-Kaya scheme ([BroKay03], [LBAK04, Sec. 4.2.2]), conditional on the drawn variance paths. The variance swap is then computed (not sampled) using the realized variance of the path and the drawn state. All random draws follow a fixed order, first variance, then stock.

**VARIANCE VIA EXACT CIR SAMPLING:** 

Given a current variance $V_k$, the next variance value is exactly drawn from the CIR transition law ([Buh01, Sec. 5.2]) 

$$
V_{k+1} = c \cdot \chi'^{2}_{\nu}(\lambda)
$$ 

where $\chi'^{2}_{\nu}(\lambda)$ denotes a noncentral chi-square random variable with $\nu$ degrees of freedom and noncentrality $\lambda$ and 

$$
\nu=\frac{4\kappa\theta_{par}}{\xi^2}
$$  

$$
c =\frac{\xi^2\bigl(1 - e^{-\kappa\mathrm{d}t}\bigr)}{4\kappa}
$$ 

$$
\lambda = \frac{4\kappa e^{-\kappa\mathrm{d}t} V_k}{\xi^2\bigl(1 - e^{-\kappa\mathrm{d}t}\bigr)}
$$ 

The factor $c$ rescales the draw into variance units, $\nu$ is fixed by the model parameters of 7.2 and $\lambda$ carries over the current level $V_k$. Every variance value is a genuine draw from the true one-step distribution, hence variance paths carry no discretization error. Since the noncentral chi-square distribution is supported on $[0, \infty)$ its square root $\sqrt{V_t}$ in the stock dynamics of 7.2 is always well defined (in contrast Euler schemes can produce negative values!).


**INTEGRATED VARIANCE VIA TRAPEZOIDAL RULE:** 

The stock and variance swap update require both the realized variance over one time step 

$$
I_k = \int_{t_k}^{t_{k+1}} V_s\mathrm{d}s \approx \frac{\mathrm{d}t}{2}\bigl(V_k + V_{k+1}\bigr)
$$ 

which is the average of the endpoint variances (only approximation in the whole scheme). Cumulating gives the realized variance up to any date, $\int_0^{t_k} V_s\mathrm{d}s = \sum_{j=0}^{k-1} I_j$.

**STOCK VIA SIMPLIFIED BROADIE-KAYA:** 

Over one step, the exact solution of the stock SDE of 7.2 (zero drift under $\mathbb{Q}$, since $r = 0$) is 

$$
\ln(S^1_{k+1}) = \ln(S^1_k) - \frac{1}{2} I_k + \int_{t_k}^{t_{k+1}} \sqrt{V_s}\mathrm{d}W^S_s
$$ 

with $W^S$ as a Wiener process. Writing $W^S = \rho W^V + \sqrt{1-\rho^2}B$ (decomposition into variance driver $W^V$ plus an independent Brownian motion $B$) the correlated part of the integral needs not to be drawn at all. Then integrating the variance SDE of 7.2 over a single step yields 

$$
\xi\int_{t_k}^{t_{k+1}} \sqrt{V_s}\mathrm{d}W^V_s = V_{k+1} - V_k - \kappa\theta_{par}\mathrm{d}t + \kappa I_k
$$ 

determined by the drawn variance change, only the orthogonal part remains random. Conditional on the variance path it is Gaussian with variance $(1-\rho^2)I_k$ and a new standard normal shock $G_k$ per step updates the stock by 

$$
\ln (S^1_{k+1}) = \ln (S^1_k) + \underbrace{\frac{\rho}{\xi} \bigl(V_{k+1} - V_k - \kappa\theta_{par}\mathrm{d}t + \kappa I_k\bigr)}_{\text{correlation term, determined by the variance path}} - \underbrace{\frac{1}{2} I_k}_{\text{Itô correction}} + \underbrace{\sqrt{(1-\rho^2) I_k} G_k}_{\text{independent noise}}
$$ 

where $G_k$ is standard normal and independent of everything drawn before, it is the only random component in the stock update. Given the variance path, it generates the stock's uncorrelated fluctuation (i.e. the component that would move the stock even if variance stayed flat).

**VARIANCE SWAP:** 

The swap paths follow from the identity of Section 7.2, with the realized part accumulated from the $I_j$ and the expected future part $L(t_k, V_k)$ evaluated at the drawn state. This makes the variance tradeable, while $S_t^2$ is a $\mathbb{Q}$-Martingale.

**OUTPUTS AND REPRODUCIBILITY:** 

One dataset consists of three matrices, stock, variance and variance swap, each with $n+1$ columns and one row per scenario, plus one payoff for each scenario. The whole dataset is generated from a single seed, so the same seed reproduces bit-identical paths, which anchors the reference values of Section 6.
### 7.5 Deep Hedging

Deep hedging solves the problem described in 7.1 by reinforcement learning techniques, where at each date $t_k$, the holdings $\delta_k$ are chosen based on observed market state (modeled by filtration $\mathbb{F} = (\mathcal{F_{k}})$ for $k=0,\dots,n$ since strategies must be $\mathcal{F}_k$-measurable) and the reward is the terminal hedging error $\varepsilon$. The optimization therefore runs over sequences of arbitrary functions which span an infinite-dimensional search space that no computation can traverse directly. Two reductions make it computable.

**REDUCTION 1, THE STATE:** 

Since $(S_t^1,V_t)$ is Markov, the optimization is restricted to depend on current state only (notice, with trading costs this no longer holds!). It therefore suffices to approximate for each rebalancing step $f_k:(s,v)\mapsto\mathbb{R}^2$, which is the mapping  of current stock and variance to the holdings.


**REDUCTION 2, DFNNS:** 

Each $f_k$ is then approximated by a DFNN $F_{\theta_k}$ with network layers $\ell = 1, \dots, L$ and using input $x_0 = (\ln(s), v)$ to compute 

$$
x_\ell = \underbrace{\sigma\bigl(A_\ell x_{\ell-1} + b_\ell\bigr)}_{\text{hidden layers}}$$ $$F_{\theta_k}(x_0) = \underbrace{A_L x_{L-1} + b_L}_{\text{linear output: } (\delta^1_k, \delta^2_k)}
$$ 

where each $A_\ell$ is a weight matrix, $b_\ell$ a bias vector and $\sigma$ a nonlinear activation applied componentwise and given by $\sigma(x) = \max \lbrace x, 0 \rbrace $. The parametrized strategy is then 

$$
\delta^{\theta}_k = F_{\theta_k}(\ln(S^1_k), V_k)
$$ 

where $\theta$ collects the weights of all $n$ networks. Substituting $\delta^{\theta}$ into the loss of 7.1 yields 

$$
J(\theta, w) = w + \frac{1}{1-\alpha} \mathbb{E}[\max\lbrace -\varepsilon(\delta^{\theta}) - w, 0\rbrace]
$$ 

with 

$$
J^\ast = \inf_{\theta, w} J(\theta, w)
$$  

This turns the optimization over the infinite-dimensional space of admissible hedging strategies to a finite-dimensional optimization over $\theta$ and $w$. This makes the problem computationally tractable.


**WHY THE REDUCTIONS ARE SOUND:** 

The DFNNs approximate the Markov strategy functions $f_k$ arbitrarily well as their capacity increases  (see [Hor91]). Therefore, the computed values for $J(\theta,w)$ converge (in the limit) to the optimal value $J^\ast$ (see [Buh01, Proposition 4.9]). Hence optimizing the DFNN parameters provides an approximation of the optimal strategy up to an arbitrarily small error value.



**HOW THE MINIMUM IS COMPUTED:** 

The expectation in $J$ is estimated on synthetic market paths (Section 7.4). It is replaced by the average over a mini-batch of $B$ scenarios randomly drawn from the (simulated) training set 

$$
J_B(\theta, w)= w + \frac{1}{1-\alpha} \underbrace{\frac{1}{B}\sum_{m=1}^{B}}_{\text{average over } B \text{ paths}} \max\lbrace -\varepsilon(\omega_m) - w, 0\rbrace
$$  

where $\omega_1, \dots, \omega_B$ are the drawn market scenarios and $\varepsilon(\omega_m)$ the terminal hedging error of strategy $\delta^{\theta}$ on scenario $\omega_m$. Because this objective is built from differentiable operations, its gradients are well defined, hence backpropagation applicable and stochastic gradient descent (Adam [KB15]) updates $\theta$ and $w$.

**What enters the training.** Only (simulated) paths for $(\log (S_t^1), V_t)$ and their resulting $\varepsilon(\delta^{\theta})$, neither pricing model nor greeks are used. The pricing machinery of 7.3 is only used for benchmarking, this means calculating the model-delta hedge against which the ML hedge is evaluated.

**What comes out.** After training, the output is the learned hedge $\delta^{\theta^\ast}$, whose loss approximates the minimal $J^\ast$ (residual risk after collecting the premium $q$). The associated risk-adjusted price is then $p_0 = q + J^\ast$ (by Proposition 3.10(ii) of [Buh01], $p_0 \ge q$).

### 7.6 Settings, synthetic data simulation and algorithm
In this section we outline **our** implementation, which includes model assumptions, parameter definition, path simulation, network architecture, computational graph design, training loop and validation. We closely follow the design in [Buh01].

**1. Assumptions**
- The market is driven by the Heston stochastic volatility model (see 7.2) under the risk-neutral measure $\mathbb{Q}$
- Variance $V_t$ is not directly tradeable, hence Variance swap $S_t^2$ is introduced
- Zero interest rate $r = 0$
- No dividends
- No transaction costs, no market impact, no liquidity constraints
- No short-selling restrictions, holdings are unbounded (linear output layer of the DFNNs, Section 7.5)
- Trading only at the discrete rebalancing dates
- The hedge portfolio is self financing after premium $q$ is collected at $t = 0$, no cash enters or leaves
- A European vanilla call is shorted, with liability $Z=\max \lbrace S^1_T-K,0\rbrace $ 

**2. Market and time parameters**

All parameters are fixed before simulation, training and validation. Defaults follow Section 5 of [Buh01].

- Initial spot at $S_0^1 = 100$
- Initial variance at $V_0 = 0.04$ 
- Strike at $K = 100$ 
- Maturity at $T = 30/365$ (30 trading days)
- Step size as $\mathrm{d}t = 1/365$ 
- The time grid is $t_k = k \mathrm{d}t$ for $k = 0, \ldots, n$ with $n = 30$, hence $T = t_n$ 
- Trading (rebalancing) dates are $t_0, \ldots, t_{n-1}$, where holdings $\delta_k$ are set at $t_k$ and held over $[t_k, t_{k+1}]$, while at $t_n = T$ no trading occurs
- Mean-reversion speed at $\kappa = 1.0$
- Long-run variance at $\theta_{par} = 0.04$ 
- Volatility of volatility at $\xi = 2.0$ 
- Correlation between stock and variance is $\rho = -0.7$ 

**3. Loss function and risk preference**
- Set $\alpha=0.5$
- The to be minimized loss function is the OCE representation of CVaR, in its parametrized form (over all network weights $\theta$ and threshold $w$) it is given by

$$
J(\theta, w) = w + 2 \mathbb{E}[\max \lbrace  -(q - Z + \sum_{k=0}^{n-1}([F_{\theta_k}(\ln S^1_k, V_k)]_1 \cdot (S^1_{k+1} - S^1_k) + [F_{\theta_k}(\ln S^1_k, V_k)]_2 \cdot (S^2_{k+1} - S^2_k))) - w, 0 \rbrace ]
$$ 

where $[\cdot]_1, [\cdot]_2$ select the first and second output component of the respective DFNN

**4. Network architecture**
- For each $k=0,\ldots,29$ exists one $F_{\theta_k}$ (DFNN), which:
    - takes $x_0=(\ln(S^1_k), V_k)$ as input (2 features)
    - consists of layers $\ell^k=(\ell^k_1,\ell^k_2,\ell^k_3, \ell^k_4)$ with architecture:
        - one input layer $\ell^k_1$ with $\dim(\ell^k_1)=2$ neurons
        - one affine map $z_1 = A^k_1 x_0 + b^k_1 \in \mathbb{R}^{17}$, followed by batch normalization
        - one hidden layer $\ell^k_2$ with $\dim(\ell^k_2)=17$ neurons, holding $x_1 = \sigma(\mathrm{BN}(z_1))$ 
        - one affine map $z_2 = A^k_2 x_1 + b^k_2 \in \mathbb{R}^{17}$, followed by batch normalization
        - one hidden layer $\ell^k_3$ with $\dim(\ell^k_3)=17$ neurons, holding $x_2 = \sigma(\mathrm{BN}(z_2))$ 
        - one output layer $\ell^k_4$ with $\dim(\ell^k_4)=2$ neurons, computed by affine mapping $A^k_3 x_2 + b^k_3$, without batch normalization and without activation
        - where componentwise applied activation $\sigma: \mathbb{R}^{17} \to \mathbb{R}^{17}$ is defined as $\sigma(x) = \max \lbrace x, 0 \rbrace$ (ReLU)
    - outputs $(\delta^1_k, \delta^2_k)$ 
- The initial OCE threshold is set to $w = 0$, shared across all $k = 0, \ldots, 29$ and trained jointly with all weights

- Batch normalization is applied componentwise to the affine pre-activations $z_1, z_2 \in \mathbb{R}^{17}$, with numerical stability constant $\epsilon = 10^{-3}$
    - for $r = 1, \ldots, 17$:
        - compute statistics over paths currently processed together via $\mu_r = \frac{1}{N}\sum_{m=1}^{N} z^m_r$ and  $s^2_r = \frac{1}{N}\sum_{m=1}^{N} \bigl(z^m_r - \mu_r\bigr)^2$
        
        - for $m = 1, \ldots, N$ apply $\mathrm{BN}(z)^m_r = \gamma_r \frac{z^m_r - \mu_r}{\sqrt{s^2_r + \epsilon}} + \beta_r$

- where $\gamma_r, \beta_r$ are learnable scale and shift, separate for each of the two normalizations, and $N$ here is the number of paths currently processed together (the mini-batch of $B=256$ during training and calibration, the full validation set during the $J_{\mathrm{val}}$ measurement).

**5. Data**
- Training set $\mathcal{D}^{\mathrm{train}}$ with $N=100000$ paths, generated with seed $s_{\mathcal{D}^{\mathrm{train}}}=15$ 
- Validation set $\mathcal{D}^{\mathrm{val}}$ with $N=100000$ paths, generated with seed $s_{\mathcal{D}^{\mathrm{val}}}=25$ 
- Out-of-sample test set $\mathcal{D}^{\mathrm{test}}$ with $N=1000000$ paths, generated with seed $s_{\mathcal{D}^{\mathrm{test}}}=35$ 
- TensorFlow seed for the initialization of the network weights: $s_{\mathrm{TF}}=42$ 

**6. Training settings**
- Adam optimizer with learning rate $0.005$ 
- Batch size $B = 256$ paths per gradient step
- Number of gradient steps: $200000$, logged every $5000$ steps
- Weights $A^k_j$ are randomly initialized (Glorot uniform, controlled by TensorFlow seed), biases start at $b^k_j=0$ and batch-normalization parameters at $\gamma_r=1$, $\beta_r=0$

**7. Premium computation** 
- Via Fang-Oosterlee pricer (see 7.3) calculate model premium $q = u(0, S_0^1, V_0)$ and save it

**8. Path simulation**
- It is $n=30$ the number of time steps, with $t_k=k\mathrm{d}t$ and $k=0,\ldots,n$, and $N$ is the number of simulated paths of the dataset currently generated (see 7.6.5)

**INITIALIZATION**:
- Set $s_{\mathcal{D}^{\mathrm{train}}}=15$ as seed for training data set 
- Set $s_{\mathcal{D}^{\mathrm{val}}}=25$ as seed for validation data set 
- Set $s_{\mathcal{D}^{\mathrm{test}}}=35$ as seed for testing data set 
- for $m=1,\ldots,N$:
    - set $V^m_0=V_0$
    - set $\ln(S^{1,m}_0)=\ln(S_0^1)$
    - set $A^m_0=0$
- set $\nu=\frac{4\kappa\theta_{par}}{\xi^2}$
- set $c=\frac{\xi^2(1-e^{-\kappa\mathrm{d}t})}{4\kappa}$

**VARIANCE**:
- for $k=0,\ldots,n-1$:
    - for $m=1,\ldots,N$:
        - compute $\lambda^m_k= \frac{4\kappa e^{-\kappa\mathrm{d}t}V^m_k} {\xi^2(1-e^{-\kappa\mathrm{d}t})}$
        - draw $V^m_{k+1}= c\chi_{\nu}^{\prime 2}(\lambda^m_k)$

**INTEGRATED VARIANCE**:
- for $k=0,\ldots,n-1$:
    - for $m=1,\ldots,N$:
        - compute $I^m_k=\frac{\mathrm{d}t}{2}(V^m_k+V^m_{k+1})$
        - accumulate $A^m_{k+1}=A^m_k+I^m_k$


**STOCK**:
- for $m=1,\ldots,N$:
    - for $k=0,\ldots,n-1$:
        - draw one shock $G^m_k\sim N(0,1)$
        - update $\ln(S_{k+1}^{1,m})=\ln(S_{k}^{1,m})+\frac{\rho}{\xi}(V_{k+1}^{m}-V_{k}^{m}-\kappa \theta_{par} \mathrm{dt}+\kappa I_{k}^{m})-\frac{1}{2}I_{k}^{m}+\sqrt{(1-\rho^{2})I_{k}^{m}}G_{k}^{m}$

**EXPONENTIATE**:
- for $k=0,\ldots,n$:
    - for $m=1,\ldots,N$:
        - set $S^{1,m}_k=e^{\ln(S^{1,m}_k)}$

**VARIANCE SWAP**:
- for $k=0,\ldots,n$:
    - for $m=1,\ldots,N$:
        - set $\tau_k=T-k\mathrm{d}t$
        - compute $L(t_k,V^m_k) = \frac{V^m_k-\theta_{par}}{\kappa} (1-e^{-\kappa\tau_k}) +\theta_{par}\tau_k$
        - compute $S^{2,m}_k=A^m_k+L(t_k,V^m_k)$

**PAYOFF**:
- for $m=1,\ldots,N$:
    - compute $Z^m=\max \lbrace S^{1,m}_n-K,0\rbrace $

**DATA PREPARATION**:
- for $m=1,\ldots,N$:
    - for $k=0,\ldots,n-1$:
        - construct $\mathcal{D}=(\ln(S^{1,m}_k), V^m_k, S^{1,m}_{k+1}-S^{1,m}_k,S^{2,m}_{k+1}-S^{2,m}_k,Z^m)$
- store resulting tensors as $\texttt{float32}$
  
**9. Model construction, training and validation**

**INITIALIZATION**:
- Set $s_{\mathrm{TF}}$ as seed for TensorFlow operations 
- for $k=0,\ldots,n-1$:
    - construct one DFNN $F_{\theta_k}$ with the architecture from step 7.6.4
    - initialize each weight matrix by Glorot uniform $A_\ell\sim U\!\left[-\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}\right]$ where $\ell\in\{1,2,3\}$ indexes the three affine maps and $n_{\mathrm{in}},n_{\mathrm{out}}$ are their input and output dimensions
    - initialize biases and batch-normalization parameters by $b_\ell=0$, $\gamma_r=1$, $\beta_r=0$
- initialize the trainable OCE threshold by $w=0$ 
- store the premium $q$ from step 7.6.7 in the model

**TRAINING**:
- initialize Adam with learning rate $\eta=0.005$
- shuffle the path indices $1,\ldots,N$ and partition them into batches of $B=256$
- for $j=1,\ldots,200000$:
    - select the next mini-batch and denote its paths by $b=1,\ldots,B$ 
    - for $k=0,\ldots,n-1$:
        - compute the holdings $(\delta^{1,b}_k,\delta^{2,b}_k)= F_{\theta_k}
          (\ln S^{1,b}_k,V^b_k)$ 
    - for $b=1,\ldots,B$:
        - compute the terminal hedging error $\varepsilon^b =q-Z^b+ \sum_{k=0}^{n-1}[\delta^{1,b}_k (S^{1,b}_{k+1}-S^{1,b}_k) + \delta^{2,b}_k (S^{2,b}_{k+1}-S^{2,b}_k)]$
        - set $L^b=-\varepsilon^b$ 
    - compute the mini-batch OCE objective $J(\theta,w)= w+\frac{1}{1-\alpha}\frac{1}{B}\sum_{b=1}^{B}\max \lbrace L^b-w,0 \rbrace $
    - update all trainable parameters by one Adam step
    - if $j=1$ or $j\equiv0\pmod{5000}$:
        - evaluate on the full validation dataset $J_{\mathrm{val}} = w_j+ \frac{1}{1-\alpha} \frac{1}{N} \sum_{m=1}^{N}\max(L^m-w_j,0)$
        - record $\bigl(j,J(\theta_j,w_j),J_{\mathrm{val}}\bigr)$ where $\theta_j,w_j$ are the parameter values after step $j$ 

**BATCH-NORMALIZATION CALIBRATION**:
- update the BN moving statistics after each training-mode forward pass by $\bar\mu_r \leftarrow 0.99\bar\mu_r+0.01\mu_r$ and $\bar s_r^2\leftarrow 0.99\bar s_r^2+0.01s_r^2$
- set the number of calibration passes to $P= \left\lceil\frac{800}{\lfloor N/B\rfloor}\right\rceil$
- for $p=1,\ldots,P$:
    - shuffle the path indices
    - process $\mathcal D^{\mathrm{train}}$ in mini-batches of size $B$ 
    - perform forward passes only (no gradients or parameter updates)
- with the default $\lfloor N/B\rfloor=390$ and $P=3$ the remaining contribution of the initial statistics is $0.99^{1173}\approx7.6\times10^{-6}$

**VALIDATION**:
- all forward passes use the calibrated moving statistics $(\bar\mu_r,\bar s_r^2)$, not batch statistics
- generate $\mathcal D^{\mathrm{test}}$ with seed $s_{\mathcal{D}^{\mathrm{test}}}=35$ using the procedure of step 7.6.8
- for $\mathcal D\in \{\mathcal D^{\mathrm{val}},\mathcal D^{\mathrm{test}}\}$
    - for $m=1,\ldots,N$:
        - compute terminal hedging error, with the ML holdings of the forward pass over $\mathcal D$: $\varepsilon^m = q-Z^m+\sum_{k=0}^{n-1} [\delta^{1,m}_k (S^{1,m}_{k+1}-S^{1,m}_k) + \delta^{2,m}_k (S^{2,m}_{k+1}-S^{2,m}_k)]$
        - set $L^m=-\varepsilon^m$
    - compute mean and standard deviation of $\{\varepsilon^m\}_{m=1}^{N}$ 
    - compute approximations of OCE objective $J^\ast = w^\ast+ \frac{1}{1-\alpha} \frac{1}{N} \sum_{m=1}^{N} \max \lbrace L^m-w^\ast,0 \rbrace $ 
    - sort the losses $L^{(1)}\ge L^{(2)}\ge\cdots\ge L^{(N)}$
    - compute empirical sorted CVaR $\widehat{\mathrm{CVaR}}_\alpha = \frac{1}{\lfloor(1-\alpha)N\rfloor} \sum_{i=1}^{\lfloor(1-\alpha)N\rfloor}L^{(i)}$
    - compute empirical VaR $\widehat{\mathrm{VaR}}_\alpha = \inf \lbrace  x: \frac{1}{N} \# \lbrace m:L^m\le x \rbrace  \ge\alpha \rbrace $
    - compare $w^\ast$ with $\widehat{\mathrm{VaR}}_\alpha$ 
    - compute the risk-adjusted price $p_0=q+J^\ast$
      
**10. Benchmarking of ML model VS analytical estimates**

**ANALYTICAL BENCHMARK** (on $\mathcal D^{\mathrm{val}}$):
- for $k=0,\ldots,n-1$:
    - set $\tau_k=T-k\mathrm{d}t$
    - for $m=1,\ldots,N$:
        - evaluate $\delta^{1,m}_k = \partial_su \bigl(t_k,S^{1,m}_k,V^m_k\bigr)$ and 
          $\delta^{2,m}_k = \frac{ \partial_vu (t_k,S^{1,m}_k,V^m_k)}{ \partial_vL (t_k,V^m_k)}$ 
- compute modelhedge error $\varepsilon^m_{\mathrm{model}}= q-Z^m+\sum_{k=0}^{n-1} [\delta^{1,m}_k (S^{1,m}_{k+1}-S^{1,m}_k) + \delta^{2,m}_k (S^{2,m}_{k+1}-S^{2,m}_k)]$
- report the mean, standard deviation and sorted $\mathrm{CVaR}_\alpha$ of $\{\varepsilon^m_{\mathrm{model}}\}_{m=1}^{N}$
  
## 8. Sources

- [Buh01] Buehler, Gonon, Teichmann, Wood: *Deep Hedging*, arXiv:1802.03042
- [LBAK04] Andersen, Jäckel, Kahl: *Simulation of square-root processes*
- [BroKay03] Broadie, Kaya: *Exact Simulation of Option Greeks under Stochastic Volatility*
- [FangOosterlee08] Fang, Oosterlee: *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions* 
- [Sey15] Seydel: Tools for Computational Finance
- [Hor91] Hornik, Stinchcombe, White: *Multilayer feedforward networks are universal approximators.*
- [KB15] Kingma, Ba: *Adam: A Method for Stochastic Optimization*
