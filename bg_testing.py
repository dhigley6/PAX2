"""
Module for testing background generation and subtraction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.signal import convolve

import LRDeconvolve
from pax_simulations import model_photoemission, run_analyze_save_load, simulate_pax
from visualize import plot_result

parameters = {
    'energy_spacing': 0.005,
    'iterations': 1E3,
    'simulations': 10,
    'cv_fold': 2,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def run_test(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    saved = run_analyze_save_load.load(log10_num_electrons, rixs, photoemission)
    plot_result.make_plot(saved['deconvolver_gs'].best_estimator_, saved['sim'])

def run_test2(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    pass

def sim_analyze_with_bg():
    pass

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
        impulse_response['y'],
        pax_spectra['x'],
        iterations=1E2,
        ground_truth_y=xray_xy['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_widths']}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.array(pax_spectra['y'])-estimated_bg)
    plot_result.make_plot(deconvolver_gs.best_estimator_, pax_spectra, xray_xy)
    
    estimated_bg = convolve(np.mean(pax_spectra['y'], axis=0), impulse_response['bg'], mode='valid')
    plt.figure()
    plt.plot(pax_spectra['x'], np.mean(pax_spectra['y'], axis=0))
    plt.plot(pax_spectra['x'], estimated_bg)
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        iterations=1E2,
        ground_truth_y=xray_xy['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_widths']}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.mean(pax_spectra['y'], axis=0)-estimated_bg)
    plot_result.make_plot(deconvolver_gs.best_estimator_, pax_spectra, xray_xy)