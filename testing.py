"""
Module for testing new ideas
"""

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

import LRDeconvolve
from pax_simulations import run_analyze_save_load
from pax_simulations import simulate_pax
from visualize import plot_photoemission, plot_result, cv_plot

PARAMETERS = {
    'energy_spacing': 0.005,
    'iterations': 1E2,
    'simulations': 1E3,
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def run_test(log10_num_electrons=10, rixs='schlappa', photoemission='ag'):
    run_analyze_save_load.run(log10_num_electrons, rixs, photoemission, **PARAMETERS)

def convergence_test2(log10_num_electrons=4, rixs='schlappa', photoemission='ag', iterations=1E3):
    run_analyze_save_load.assess_convergence(log10_num_electrons, rixs, photoemission)

def tf_train_step(deconvolved, optimizer):
    with tf.GradientTape() as tape:
        reconvolved = tf.signal.convolve()
        loss = mean_squared_error(measurement, reconvolved)
    grads = tape.gradient(loss, deconvolved)
    optimizer.apply_gradients(zip(deconvolved, last))