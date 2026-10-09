# test_reference_values.py
"""Reference-value tests for the deep-hedging pipeline.

All values bind to the SHIPPED checkpoint model ml_models/heston_30d_alpha_50
(paper defaults, 200k steps). Loading is deterministic: no training, no
calibration. Run from anywhere:  pytest tests/
"""
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = str(PROJECT_ROOT / "ml_models")
MODEL_NAME = "heston_30d_alpha_50"

# Fixed reference values, Section 3 of the project document.
Q_EXPECTED = 1.6918355192192847
VAL_J_STAR = 0.2774839103
VAL_P0 = 1.9693194330
MODEL_HEDGE_STD = 0.38078850095234296
MODEL_HEDGE_CVAR = 0.2433
EXP_FORMULA = -1.0672226722144846
EXP_ML = -1.0271626533897824

@pytest.fixture(scope="session")
def loaded():
    """Load the shipped model once per test run (TF import is the slow part)."""
    import os
    os.chdir(PROJECT_ROOT)
    from deep_hedging_heston import DeepHedger
    dh = DeepHedger.load_model(MODEL_DIR, MODEL_NAME)

    from deep_hedging_heston.simulate import simulate_market
    from deep_hedging_heston.modeldesign import to_tf
    from deep_hedging_heston.evaluation import HedgeEvaluation

    hp = dh.heston_parameters
    val_paths = simulate_market(hp, seed=25, nr_of_simulated_paths=100_000)
    val_tf = to_tf(val_paths)
    ev = HedgeEvaluation(dh.model, hp)

    result = ev.evaluate(val_tf, name="val")
    model_errors = ev.model_hedge_errors(val_paths, hp)
    losses = -model_errors
    model_cvar = float(np.sort(losses)[::-1][:int(0.5 * len(losses))].mean())

    return dh, hp, result, model_errors, model_cvar


def test_premium_from_pricer(loaded):
    """q must be bit-identical (0 tolerance)."""
    _, _, _, _, _ = loaded
    from deep_hedging_heston.parameters import HestonParameters
    from deep_hedging_heston.pricers import heston_call_deltas

    hp = HestonParameters()
    price, _, _ = heston_call_deltas(hp.S0, hp.v0, hp.T, hp)
    assert price.item() == Q_EXPECTED


def test_val_report(loaded):
    """J* and p0 of the shipped model on the val set (1e-6)."""
    _, _, result, _, _ = loaded
    assert abs(float(result["J_star"]) - VAL_J_STAR) < 1e-6
    assert abs(float(result["p0"]) - VAL_P0) < 1e-6


def test_model_hedge_bit_identical(loaded):
    """Model hedge uses no networks: std must be bit-identical, CVaR ~1e-4 (metric)."""
    _, _, _, model_errors, model_cvar = loaded
    assert model_errors.std().item() == MODEL_HEDGE_STD
    assert abs(model_cvar - MODEL_HEDGE_CVAR) < 1e-4


def test_experiment_seed4(loaded):
    """Single-path experiment: formula bit-identical (0 tolerance), ML 1e-6."""
    dh, _, _, _, _ = loaded
    out = dh.experiment(seed=4)
    eps_formula = out["eps_formula"]
    eps_ml = out["eps_ml"]
    eps_formula = eps_formula.item() if hasattr(eps_formula, "item") else eps_formula
    eps_ml = eps_ml.item() if hasattr(eps_ml, "item") else eps_ml
    assert eps_formula == EXP_FORMULA
    assert abs(eps_ml - EXP_ML) < 1e-6