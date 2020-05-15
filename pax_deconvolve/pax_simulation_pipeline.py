"""
Pipeline for running, analyzing, saving, and loading PAX simulations.
"""

import numpy as np
import os
import pickle
from sklearn.model_selection import GridSearchCV
import pprint
from joblib import Parallel, delayed

from pax_deconvolve.deconvolution import deconvolvers
from pax_deconvolve.pax_simulations import simulate_pax

# Set global simulation parameters
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'simulated_results')
# Set default simulation parameters
DEFAULT_PARAMETERS = {
    'energy_loss': np.arange(-8, 10, 0.005),
    'iterations': 1E3,
    'simulations': 1000,
    'cv_fold': 4,
    'regularizer_widths': np.logspace(-3, -1, 10)
}
# good energy_loss for Ag 3d levels with Schlappa RIXS: np.arange(-8, 10, 0.005)
# good energy_loss for Fermi edge and doublet with < 0.4 eV separation: np.arange(-0.5, 0.5, 0.001)
# good regularizer_widths for Ag 3d: np.logspace(-3, -1, 10)
# good regularizer_widths for Fermi edge: np.logspace(-4, -2, 10)


def run_with_extra(log10_num_electrons, rixs='schlappa', photoemission='ag', num_analyses=25, **kwargs):
    parameters = DEFAULT_PARAMETERS
    parameters.update(kwargs)
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        rixs,
        photoemission,
        parameters['simulations'],
        parameters['energy_loss']
    )
    deconvolver = deconvolvers.LRFisterGrid(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        parameters['regularizer_widths'],
        parameters['iterations'],
        xray_xy['y'],
        parameters['cv_fold']
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    regularizer_width = deconvolver.best_params_['regularizer_width']
    analyses = Parallel(n_jobs=-1)(delayed(_run_single_analysis)(log10_num_electrons, rixs, photoemission, regularizer_width, parameters) for _ in range(num_analyses))
    to_save = {
        'cv_deconvolver': deconvolver,
        'analyses': analyses,
        'pax_spectra': pax_spectra
    }
    file_name = _get_filename_with_extra(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'wb') as f:
        pickle.dump(to_save, f)
    return to_save


def _run_single_analysis(log10_num_electrons, rixs, photoemission, regularizer_width, parameters):
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        rixs,
        photoemission,
        parameters['simulations'],
        parameters['energy_loss']
    )
    deconvolver = deconvolvers.LRFisterDeconvolve(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        regularizer_width,
        iterations=parameters['iterations'],
        ground_truth_y=xray_xy['y']
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    return deconvolver

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
        parameters['energy_loss']
    )
    deconvolver = deconvolvers.LRFisterGrid(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        parameters['regularizer_widths'],
        parameters['iterations'],
        xray_xy['y'],
        parameters['cv_fold']
    )
    deconvolver.fit(np.array(pax_spectra['y']))
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

def load_with_extra(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    """Load PAX simulation results
    """
    file_name = _get_filename_with_extra(log10_num_electrons, rixs, photoemission)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def print_parameters(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    """Load a PAX simulation and print some parameters it was run with
    """
    data = load(log10_num_electrons, rixs, photoemission)
    to_print = {
        'iterations': data['deconvolver'].iterations,
        'cv_fold': data['deconvolver'].cv_,
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

def _get_filename_with_extra(log10_num_electrons, rixs, photoemission):
    file_name = '{}/{}_{}_rixs_1E{}_with_extra.pickle'.format(
        PROCESSED_DATA_DIR,
        photoemission,
        rixs,
        log10_num_electrons
    )
    return file_name