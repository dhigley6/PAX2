"""
Module for testing new ideas
"""

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from pax_simulations import run_analyze_save_load

PARAMETERS = {
    'energy_spacing': 0.005,
    'iterations': 1E2,
    'simulations': 1E3,
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def run_test(log10_num_electrons=10, rixs='schlappa', photoemission='ag'):
    run_analyze_save_load.run(log10_num_electrons, rixs, photoemission, **PARAMETERS)

def tf_train_step(deconvolved, optimizer):
    with tf.GradientTape() as tape:
        reconvolved = tf.signal.convolve()
        loss = mean_squared_error(measurement, reconvolved)
    grads = tape.gradient(loss, deconvolved)
    optimizer.apply_gradients(zip(deconvolved, last))