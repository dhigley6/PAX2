"""
Module for running, analyzing, saving, and loading PAX simulations.

This module should be used as the main interaction point for doing and loading PAX simulations.
"""

import numpy as np
import os
import pickle
from sklearn.model_selection import GridSearchCV

from pax_simulations import simulate_pax
import LRDeconvolve

# Set global simulation parameters
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../simulated_results')
# Set default simulation parameters
DEFAULT_PARAMETERS = {
    'energy_spacing': 0.005,
    'iterations': 1E3,
    'simulations': 4,
    'cv_fold': 4,
    'regularizer_width': np.logspace(-3, -1, 10)
}

def run(log10_num_electrons, rixs='schlappa', photoemission='ag', **kwargs):
    """Run PAX simulation, deconvolve, then pickle the results
    """
    parameters = DEFAULT_PARAMETERS
    parameters.update(kwargs)
    sim = simulate_pax.simulate_set_from_presets(
        log10_num_electrons,
        rixs,
        photoemission,
        parameters['simulations'],
        parameters['energy_spacing']
    )
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        sim['impulse_response']['x'],
        sim['impulse_response']['y'],
        sim['pax_x'],
        iterations=parameters['iterations'],
        ground_truth_y=sim['xray_xy']['y']
    )
    param_grid = {'regularizer_width': parameters['regularizer_width']}
    deconvolver_gs = GridSearchCV(deconvolver, param_grid, cv=parameters['cv_fold'], return_train_score=True, verbose=1, n_jobs=-1)
    deconvolver_gs.fit(np.array(sim['pax_y_list']))
    to_save = {'deconvolver_gs': deconvolver_gs, 'sim': sim}
    file_name = _get_filename(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'wb') as f:
        pickle.dump(to_save, f)
    return to_save

def load(log10_num_electrons, rixs='schlappa', photoemission='ag', **kwargs):
    """Load PAX simulation results
    """
    file_name = _get_filename(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def _get_filename(log10_num_electrons, rixs, photoemission):
    file_name = '{}/{}_{}_rixs_1E{}.pickle'.format(
        PROCESSED_DATA_DIR,
        photoemission,
        rixs,
        log10_num_electrons)
    return file_name