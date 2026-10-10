# orchestrate.py
import os
import json
import tensorflow as tf # to save the ML model
from .parameters    import HestonParameters, TrainingParameters, DFNNParameters
from .pricers       import heston_call_deltas
from .simulate      import simulate_market
from .modeldesign   import DeepHedgingModel, reset_tensorflow, to_tf, DTYPE
from .modeltraining import DeepHedgingTrainer
from .evaluation    import HedgeEvaluation
from .experiment    import run_experiment

class DeepHedger:
    """Entry point of the package, gives you the option to build, train, save, load and evaluate a
    deep hedging model for a European call in a Heston market.

    - DeepHedger(...)                            instantiates the DeepHedger class
    - .run()                                     allows you to train and evaluate a model
    - .save_model(folderpath, name_of_ML_model)  allows you to save a trained model with name name_of_ML_model to folderpath
    - DeepHedger.load_model(...)                 allows you to load a saved model, ready to use
    - .experiment(seed)                          allows you to build a new market scenario and check the ML hedge vs analytical benchmark
    """
    
    def __init__(self, training_parameters=None, heston_parameters=None, dfnn_parameters=None):
        """ All three arguments are optional: DeepHedger() with no argument gives you the paper setting of the shipped model heston_30d_alpha_50.
        - heston_parameters:   HestonParameters object, sets the market, the strike and time grid
        - training_parameters: TrainingParameters object, sets the CVaR level alpha, Adam settings, number of paths and all seeds
        - dfnn_parameters:     DFNNParameters object, sets the hidden width, activation and startvalue of w
        """
        
        self.training_parameters = training_parameters or TrainingParameters()
        self.heston_parameters = heston_parameters or HestonParameters()
        self.dfnn_parameters = dfnn_parameters or DFNNParameters()
        premium_array, _, _ = heston_call_deltas(
            self.heston_parameters.S0, self.heston_parameters.v0, self.heston_parameters.T,
            self.heston_parameters)
        self.premium = float(premium_array[0])                                                  # q (premium) — the only market output the ML needs

    def run(self):
        """Allows you to build and train a new model: simulates the market paths,
        trains all networks and the OCE threshold w, calibrates batch normalization and
        evaluates the hedge on validation data, out-of-sample data and against the
        analytical benchmark.

        Takes no arguments, all settings come from the constructor. Returns and stores a
        results dictionary: self.results with keys 'premium', 'result_val', 'result_oos',
        'deep_result', 'model_result'.
        """
        
        training_parameters = self.training_parameters

        reset_tensorflow(training_parameters.seed_tf)                                           # clean session + reproducible initial weights
        model = DeepHedgingModel(self.heston_parameters, self.dfnn_parameters,
                                 self.premium, training_parameters)                             # create the model (untrained)

        training_paths = simulate_market(self.heston_parameters, seed=training_parameters.seed_train,
                                         nr_of_simulated_paths=training_parameters.paths)

        # Validation paths BEFORE training, the basis of the J(val) curve and later of the val evaluation
        validation_paths = simulate_market(self.heston_parameters, seed=training_parameters.seed_val,
                                           nr_of_simulated_paths=training_parameters.paths)
        validation_data_tf = to_tf(validation_paths)

        trainer = DeepHedgingTrainer(model, to_tf(training_paths), training_parameters,
                                     validation_data_tf=validation_data_tf)
        history = trainer.train()                                                               # training + the ONE automatic BN calibration (inside the trainer)

        evaluation = HedgeEvaluation(model, self.heston_parameters)

        result_val = evaluation.evaluate(validation_data_tf, name="val")                        # same data as in the curve

        test_paths = simulate_market(self.heston_parameters, seed=training_parameters.seed_test,
                                     nr_of_simulated_paths=training_parameters.paths_test)
        result_oos = evaluation.evaluate(to_tf(test_paths), name="OOS")

        deep_result, model_result = evaluation.compare_with_model_hedge(
            validation_data_tf, validation_paths, self.heston_parameters, name="val")

        self.model = model                                                                      # the trained model — for plots/saving
        self.history = history                                                                  # learning curve (step, J_train, J_val)
        model.train_history = history                                                           # on the model: for the experiment chart
        self.evaluation = evaluation                                                            # for further evaluations
        self.results = dict(premium=self.premium, result_val=result_val, result_oos=result_oos,
                            deep_result=deep_result, model_result=model_result)
        return self.results

    def experiment(self, seed):
        """Allows you to build a new market scenario and check its performance by comparing 
        ML hedge vs the analytical benchmark: simulates one path, hedges it with the ML model
        and also with the analytical benchmark and plots 4x2 charts (market, hedge positions, 
        PnL/portfolio, terminalerrors, learning curve).

        - seed: must be non-negative integer, same seed gives same scenario, seed=4 reproduces the
          reference result (terminal errors -1.0672 analytical / -1.0272 ML).
        
        - Returns a dict with 'figure', 'eps_formula' and 'eps_ml' (also printed).
        
        - Requires run() or load_model() first.
        """
        
        if not hasattr(self, 'model'):
            raise RuntimeError("No model available — run run() (train) or load_model() (load) first.")
        return run_experiment(self.model, self.heston_parameters, seed)

    def save_model(self, directory, name):
        """Allows you to save a trained model with specifying its name by name and the path to save the 
        model by folderpath. writes the parameter file <name>_parameters.json (settings and learning curve) 
        and the checkpoint files <name>-1.* (weights, w, batch norm statistics).

        - directory: folder path, created if it does not exist, e.g. "ml_models"

        - name: name of the ML model, convention 'heston_<days>d_alpha_<alpha>', e.g. "heston_15d_alpha_90"
        
        - Requires run() first.
        """
        
        os.makedirs(directory, exist_ok=True)                                                   # FIRST: make sure the folder exists
        history = getattr(self, 'history', None) or []                                          # curve; empty if run() never ran
        parameters_json = dict(                                                                 # THEN: parameter JSON
            v0=self.heston_parameters.v0, S0=self.heston_parameters.S0,
            K=self.heston_parameters.K, T=self.heston_parameters.T,
            timesteps=self.heston_parameters.timesteps,
            rho=self.heston_parameters.rho, kappa=self.heston_parameters.kappa,
            theta=self.heston_parameters.theta, xi=self.heston_parameters.xi,
            ALPHA=self.training_parameters.ALPHA,
            nr_of_units=self.dfnn_parameters.nr_of_units,
            activation=self.dfnn_parameters.activation,
            history=[[float(step), float(j_train), float(j_val)] for (step, j_train, j_val) in history])
        with open(directory + "/" + name + "_parameters.json", "w") as file:
            json.dump(parameters_json, file, indent=2)
        checkpoint = tf.train.Checkpoint(networks=self.model.networks, oce_threshold=self.model.oce_threshold)
        checkpoint.save(directory + "/" + name)

    @classmethod
    def load_model(cls, directory, name):
        """Allows you to load a saved model to experiment with it: reads the files written by
        save_model(), rebuilds the architecture, restores the weights and recomputes the
        premium q.
        
        - directory: folder path containing the files, e.g. "ml_models"
        
        - name: name of the ML model, e.g. "heston_30d_alpha_50"
        
        - Returns a ready-to-use DeepHedger, call .experiment(seed) directly.
        """
        
        with open(directory + "/" + name + "_parameters.json") as file:
            p = json.load(file)
        heston = HestonParameters(v0=p['v0'], 
                                  S0=p['S0'], 
                                  K=p['K'], 
                                  T=p['T'],
                                  timesteps=p['timesteps'], 
                                  rho=p['rho'],
                                  kappa=p['kappa'], 
                                  theta=p['theta'], 
                                  xi=p['xi'])
        training = TrainingParameters(ALPHA=p['ALPHA'])
        dfnn = DFNNParameters(nr_of_units=p['nr_of_units'], activation=p['activation'])
        reset_tensorflow(0)                                                                     # build the architecture; weights are overwritten right away
        model = DeepHedgingModel(heston, dfnn, premium=0.0, training_parameters=training)
        checkpoint = tf.train.Checkpoint(networks=model.networks, oce_threshold=model.oce_threshold)
        checkpoint.restore(directory + "/" + name + "-1").expect_partial()
        premium_array, _, _ = heston_call_deltas(heston.S0, heston.v0, heston.T, heston)
        premium = float(premium_array[0])                                                       # q NOW, before the model is used
        model.premium = tf.constant(premium, dtype=DTYPE)                                       # into the model instead of 0
        dh = cls(training, heston, dfnn)
        dh.model = model
        dh.premium = premium
        dh.history = [tuple(point) for point in p.get('history', [])]                           # learning curve (or [] for old models)
        model.train_history = dh.history                                                        # on the model: for the experiment chart
        return dh
