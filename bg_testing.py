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
    sim = simulate_pax.simulate_set_from_presets(
        log10_num_electrons,
        rixs,
        'ag_with_bg',
        4,
        0.005
    )
    plt.close('all')
    plt.figure()
    plt.plot(sim['impulse_response']['x'], sim['impulse_response']['y'])
    plt.plot(sim['impulse_response']['x'], sim['impulse_response']['y']-sim['impulse_response_bg'])
    estimated_bg = convolve(sim['mean_pax_xy']['y'], sim['impulse_response_bg'], mode='valid')
    plt.figure()
    plt.plot(sim['mean_pax_xy']['x'], sim['mean_pax_xy']['y'])
    plt.plot(sim['mean_pax_xy']['x'], estimated_bg)
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        sim['impulse_response']['x'],
        sim['impulse_response']['y']-sim['impulse_response_bg'],
        sim['pax_x'],
        iterations=1E2,
        ground_truth_y=sim['xray_xy']['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_widths']}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.array(sim['pax_y_list'])-estimated_bg)
    plot_result.make_plot(deconvolver_gs.best_estimator_, sim)
    
    estimated_bg = convolve(sim['mean_pax_xy']['y'], sim['impulse_response_bg'], mode='valid')
    plt.figure()
    plt.plot(sim['mean_pax_xy']['x'], sim['mean_pax_xy']['y'])
    plt.plot(sim['mean_pax_xy']['x'], estimated_bg)
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        sim['impulse_response']['x'],
        sim['impulse_response']['y']-sim['impulse_response_bg'],
        sim['pax_x'],
        iterations=1E2,
        ground_truth_y=sim['xray_xy']['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_widths']}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.array(sim['pax_y_list'])-estimated_bg)
    plot_result.make_plot(deconvolver_gs.best_estimator_, sim)