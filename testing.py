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
    'iterations': 1E5,
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

def run_cv_analysis(iterations=1E5):
    #points = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    points = [7.0]
    for log10_num_electrons in points:
        run_analyze_save_load.run(log10_num_electrons, rixs='schlappa', photoemission='ag', **PARAMETERS)