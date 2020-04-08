"""
Module for testing background generation and subtraction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve
import tensorflow as tf

import LRDeconvolve
from pax_simulations import model_photoemission, run_analyze_save_load, simulate_pax
from visualize import plot_result, cv_plot

parameters = {
    'energy_spacing': 0.005,
    'iterations': 1E3,
    'simulations': 10,
    'cv_fold': 2,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def test_tf(log10_num_electrons=6):
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        'schlappa',
        'ag',
        4,
        0.005
    )
    pax_average = np.mean(pax_spectra['y'], axis=0)
    adam = tf.keras.optimizers.Adam()


def tf_train_step(deconvolved, optimizer):
    with tf.GradientTape() as tape:
        reconvolved = 
        loss = mean_squared_error(measurement, reconvolved)
    grads = tape.gradient(loss, deconvolved)
    optimizer.apply_gradients(zip(deconvolved, last))

def run_test(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    saved = run_analyze_save_load.load(log10_num_electrons, rixs, photoemission)
    plot_result.make_plot(saved['deconvolver_gs'].best_estimator_)

def run_test2(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    pass

def sim_analyze_with_bg():
    pass

def test3(log10_num_electrons=6, rixs='schlappa'):
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        'schlappa',
        'ag',
        4,
        0.005
    )
    deconvolver = LRDeconvolve.LRFisterGrid(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        parameters['regularizer_widths'],
        ground_truth_y=xray_xy['y']
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    plot_result.make_plot(deconvolver)
    cv_plot.make_plot(deconvolver)


def test2(log10_num_electrons, rixs='schlappa'):
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        rixs,
        'ag_with_bg',
        4,
        0.005
    )
    plt.close('all')
    plt.figure()
    plt.plot(impulse_response['x'], impulse_response['y'])
    plt.plot(impulse_response['x'], impulse_response['y']-impulse_response['bg'])
    estimated_bg = convolve(np.mean(pax_spectra['y'], axis=0), impulse_response['bg'], mode='valid')
    plt.figure()
    plt.plot(pax_spectra['x'], np.mean(pax_spectra['y'], axis=0))
    plt.plot(pax_spectra['x'], estimated_bg)
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        impulse_response['x'],
        (impulse_response['y']-impulse_response['bg'])/np.sum(impulse_response['y']-impulse_response['bg']),
        pax_spectra['x'],
        iterations=1E3,
        ground_truth_y=xray_xy['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_widths'],
                  'ground_truth_y': [xray_xy['y']]}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.array(pax_spectra['y']))
    plot_result.make_plot(deconvolver_gs.best_estimator_)
    
    deconvolved_y_list = []
    deconvolved_y = deconvolver_gs.best_estimator_.deconvolved_y_
    deconvolved_y_list.append(deconvolved_y)
    for _ in range(3):
        deconvolved_y = georgi_bg_estimate_block(deconvolved_y, impulse_response, pax_spectra, xray_xy)
        deconvolved_y_list.append(deconvolved_y)
    plt.figure()
    plt.plot(deconvolver_gs.best_estimator_.deconvolved_x, deconvolver_gs.best_estimator_.ground_truth_y, label='Ground Truth')
    for ind, deconvolved_y in enumerate(deconvolved_y_list):
        plt.plot(deconvolver_gs.best_estimator_.deconvolved_x, deconvolved_y, label=str(ind))
        plt.legend(loc='best')

def georgi_bg_estimate_block(last_deconvolved, impulse_response, pax_spectra, xray_xy):
    estimated_bg = convolve(last_deconvolved, impulse_response['bg'], mode='valid')
    plt.figure()
    plt.plot(pax_spectra['x'], np.mean(pax_spectra['y'], axis=0))
    plt.plot(pax_spectra['x'], estimated_bg)
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        impulse_response['x'],
        (impulse_response['y']-impulse_response['bg'])/np.sum(impulse_response['y']-impulse_response['bg']),
        pax_spectra['x'],
        iterations=1E2,
        ground_truth_y=xray_xy['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_widths']}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.array(pax_spectra['y'])-estimated_bg)
    plot_result.make_plot(deconvolver_gs.best_estimator_)
    return deconvolver_gs.best_estimator_.deconvolved_y_

def renormalize_data():
    pass