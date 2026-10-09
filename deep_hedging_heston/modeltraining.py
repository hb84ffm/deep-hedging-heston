# modeltraining.py
import time                                                                                                 # measure training duration
import numpy as np                                                                                          # index stream for the batches
import tensorflow as tf                                                                                     # GradientTape, Dataset
from tensorflow import keras                                                                                # Adam


class DeepHedgingTrainer:
    """Executes a DeepHedgingModel: 
    - Adam gradient steps on the OCE loss (Eq. 4.6) plus the final BatchNorm calibration. 
    - Every PRINT_EVERY steps the OCE loss is additionally measured on the full validation 
    set and recorded as (step, J_train, J_val) in train_history
     - The data behind the learning curve in the experiment chart. 
     - The constructor does NOT train."""

    def __init__(self, deep_hedging_model, training_data_tf, training_parameters, validation_data_tf=None):
        """Sets up the tools: 
        - Adam, the cached variable list (needed for the graph), and an endless shuffled batch stream.
        - validation_data_tf: dict from to_tf() of the validation paths (for the
        J(val) curve); None is allowed (no J(val) measured) — orchestrate.run()
        always passes it."""
        self.model = deep_hedging_model                                                                    # weights change INSIDE this object
        self.data = training_data_tf                                                                       # TF training data
        self.parameters = training_parameters
        self.validation_data = validation_data_tf                                                          # TF validation data (or None)
        self.optimizer = keras.optimizers.Adam(learning_rate=training_parameters.LEARNING_RATE)
        self.trainable_variables = deep_hedging_model.collect_trainable_variables()                        # 301 = w + 30 networks x 10
        self.number_of_paths = training_data_tf['log_stocks'].shape[0]
        self.batch_stream = (tf.data.Dataset.from_tensor_slices(np.arange(self.number_of_paths))
                             .shuffle(self.number_of_paths, reshuffle_each_iteration=True)
                             .repeat()
                             .batch(training_parameters.BATCH))
        self.train_history = []                                                                            # (step, J_train, J_val) for the learning curve

    def gather_batch(self, selected_samples):
        """Gather the rows of the current batch from the training data."""
        return dict(
            log_stocks=tf.gather(self.data['log_stocks'], selected_samples),
            variance=tf.gather(self.data['variance'], selected_samples),
            stocks_increment=tf.gather(self.data['stocks_increment'], selected_samples),
            variance_increment=tf.gather(self.data['variance_increment'], selected_samples),
            payoff=tf.gather(self.data['payoff'], selected_samples))

    @tf.function
    def train_step(self, selected_samples):
        """ONE gradient step: forward pass -> OCE loss -> gradients -> Adam update."""
        data_batch = self.gather_batch(selected_samples)
        with tf.GradientTape() as tape:
            losses = self.model.compute_losses(data_batch, training=True)
            objective = self.model.compute_oce_loss(losses)
        gradients = tape.gradient(objective, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return objective

    @tf.function
    def validation_loss(self):
        """J on the FULL validation set, deliberately measured with training=True.
        - During training the BN moving statistics are UNCALIBRATED — a measurement
        with training=False would track their growing mismatch (the rebound
        artefact), not the model. 
        - training=True uses the statistics of the validation data itself = the same 
        regime as J(batch), so the curve is truly comparable. 
        - No gradients: the BN statistics updates made here are washed out by the final 
        calibration down to a remainder of ~7.6e-6 (= 0.99^1173), the accepted fingerprint, 
        documented in the README."""
        losses = self.model.compute_terminal_hedge_error(self.validation_data, training=True)
        return self.model.compute_oce_loss(losses)

    def calibrate_batchnorm(self, number_of_passes=None):
        """Finalize the BN moving statistics: forward passes over all training
        data with training=True. No gradient steps — only the moving statistics
        get their final values. This is the ONE calibration point of the
        pipeline, called automatically at the end of train().

        number_of_passes=None (default): automatic choice — passes such that
        batches x passes >= ~800, i.e. the remaining share of the initial
        statistics is < 0.05% (momentum 0.99 lets each batch learn only 1%:
        remaining share = 0.99^(batches x passes)). Full runs (390 full batches):
        ceil(800/390) = 3 passes = 1173 batches. An explicit number can still
        be passed for experiments."""
        if number_of_passes is None:
            number_of_batches = max(1, self.number_of_paths // self.parameters.BATCH)
            number_of_passes = max(1, int(np.ceil(800 / number_of_batches)))
        for _ in range(number_of_passes):
            calibration_stream = (tf.data.Dataset.from_tensor_slices(np.arange(self.number_of_paths))
                                  .shuffle(self.number_of_paths, reshuffle_each_iteration=True)
                                  .batch(self.parameters.BATCH))
            for selected_samples in calibration_stream:
                _ = self.model.compute_terminal_hedge_error(self.gather_batch(selected_samples), training=True)

    def train(self):
        """Run N_STEPS gradient steps with the log + J(val) rhythm, then calibrate
        BatchNorm automatically. 
        - The last J(val) point is measured in the batch regime and lies close to the final reported value
         -  small deviations (tenths) are normal, not a bug."""
        start_time = time.time()
        for step, selected_samples in enumerate(self.batch_stream.take(self.parameters.N_STEPS), start=1):
            objective = self.train_step(selected_samples)
            if step == 1 or step % self.parameters.PRINT_EVERY == 0:
                j_val = float(self.validation_loss()) if self.validation_data is not None else float('nan')
                self.train_history.append((step, float(objective), j_val))
                print(f"Step {step:6d}/{self.parameters.N_STEPS}   J(Batch) = {float(objective):8.4f}   "
                      f"J(val) = {j_val:8.4f}   w = {float(self.model.oce_threshold):7.4f}   "
                      f"({(time.time()-start_time)/60:5.1f}min)")
        self.calibrate_batchnorm()
        print(f"Training done in {(time.time()-start_time)/60:.1f} min — BatchNorm calibrated.")
        return self.train_history