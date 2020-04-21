"""
Module for running, analyzing, saving, and loading PAX simulations.

This module should be used as the main interaction point for doing and loading PAX simulations.
"""

import numpy as np
import os
import pickle
from sklearn.model_selection import GridSearchCV
import pprint

from pax_simulations import simulate_pax
import LRDeconvolve
from visualize import plot_photoemission, plot_result, cv_plot

# Set global simulation parameters
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../simulated_results')
# Set default simulation parameters
DEFAULT_PARAMETERS = {
    'energy_spacing': 0.005,
    'iterations': 1E3,
    'simulations': 1000,
    'cv_fold': 4,
    'regularizer_widths': np.logspace(-3, -1, 10)
}


def run(log10_num_electrons, rixs='schlappa', photoemission='ag', **kwargs):
    """Run PAX simulation, deconvolve, then pickle the results
    """
    parameters = DEFAULT_PARAMETERS
    parameters.update(kwargs)
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        rixs,
        photoemission,
        parameters['simulations'],
        parameters['energy_spacing']
    )
    deconvolver = LRDeconvolve.LRFisterGrid(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        parameters['regularizer_widths'],
        parameters['iterations'],
        xray_xy['y'],
        parameters['cv_fold']
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    plot_photoemission.make_plot(deconvolver)
    cv_plot.make_plot(deconvolver)
    plot_result.make_plot(deconvolver)
    to_save = {
        'deconvolver': deconvolver,
        'pax_spectra': pax_spectra
        }
    file_name = _get_filename(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'wb') as f:
        pickle.dump(to_save, f)
    return to_save

def load(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    """Load PAX simulation results
    """
    file_name = _get_filename(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def print_parameters(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    """Load a PAX simulation and print some parameters it was run with
    """
    data = load(log10_num_electrons, rixs, photoemission)
    to_print = {
        'iterations': data['deconvolver'].iterations,
        'cv_fold': data['deconvolver'].cv,
        'regularizer_widths': data['deconvolver'].regularizer_widths,
        'shape of input PAX data': np.shape(data['pax_spectra']['y'])
    }
    pprint.pprint(to_print)

def _get_filename(log10_num_electrons, rixs, photoemission):
    file_name = '{}/{}_{}_rixs_1E{}.pickle'.format(
        PROCESSED_DATA_DIR,
        photoemission,
        rixs,
        log10_num_electrons)
    return file_name