"""
Module for testing new ideas
"""

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import mean_squared_error

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

def convergence_test(log10_num_electrons=4, rixs='schlappa', photoemission='ag'):
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        rixs,
        photoemission,
        PARAMETERS['simulations'],
        PARAMETERS['energy_spacing']
    )
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        regularizer_width=0.005,
        iterations=1E4,
        ground_truth_y=xray_xy['y'],
        logging=True
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    plot_photoemission.make_plot(deconvolver)
    plot_result.make_plot(deconvolver)
    #deconvolver = LRDeconvolve.LRFisterGrid(
    #    impulse_response['x'],
    #    impulse_response['y'],
    #    pax_spectra['x'],
    #    PARAMETERS['regularizer_widths'],
    #    PARAMETERS['iterations'],
    #    xray_xy['y'],
    #    PARAMETERS['cv_fold']
    #)
    deconvolver = LRDeconvolve.LRDeconvolve(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        iterations=1E4,
        ground_truth_y=xray_xy['y'],
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    plot_photoemission.make_plot(deconvolver)
    plot_result.make_plot(deconvolver)

def tf_train_step(deconvolved, optimizer):
    with tf.GradientTape() as tape:
        reconvolved = tf.signal.convolve()
        loss = mean_squared_error(measurement, reconvolved)
    grads = tape.gradient(loss, deconvolved)
    optimizer.apply_gradients(zip(deconvolved, last))