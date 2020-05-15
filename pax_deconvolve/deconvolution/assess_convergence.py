"""Tool for assessing convergence of deconvolution

The intended use of this tool is to run it with the 'run' or 'run_pax_preset' functions
below, then view the results using tensorboard on the log directory.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from pax_deconvolve.deconvolution import deconvolvers
from pax_deconvolve.pax_simulations import simulate_pax
from pax_deconvolve import pax_simulation_pipeline
DEFAULT_PARAMETERS = pax_simulation_pipeline.DEFAULT_PARAMETERS

def run_pax_preset(log10_num_electrons, rixs='schlappa', photoemission='ag', **kwargs):
    """Log deconvolution results for preset PAX parameters
    To be used to make sure deconvolutions have been run for sufficient iterations.
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
    regularizer_widths = parameters['regularizer_widths']
    regularizer_widths = np.append([0], regularizer_widths)
    run(impulse_response, pax_spectra, xray_xy, regularizer_widths, parameters['iterations'])

def run(impulse_response, pax_spectra, xray_xy, regularizer_widths, iterations):
    """Log deconvolution results as a function of iteration number using tensorboard
    To be used to make sure deconvolutions have been run for sufficient iterations.
    """
    pax_spectra_train, val_pax_y = _split_pax_data(pax_spectra)
    Parallel(n_jobs=-1)(delayed(_run_single_deconvolver)(impulse_response, pax_spectra_train, xray_xy, regularizer_width, iterations, val_pax_y) for regularizer_width in regularizer_widths)
    # below code can be used for debugging in case parallel case doesn't work
    #_ = (run_single_deconvolver(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations, val_pax_y) for regularizer_width in regularizer_widths)

def _split_pax_data(pax_spectra):
    """Split PAX data into a training and validation set
    """
    pax_spectra_train_y, pax_spectra_validation_y = train_test_split(pax_spectra['y'], test_size=0.3)
    pax_spectra_train = {
        'x': pax_spectra['x'],
        'y': pax_spectra_train_y
    }
    val_pax_y = np.mean(pax_spectra_validation_y, axis=0)
    return pax_spectra_train, val_pax_y

def _run_single_deconvolver(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations, val_pax_y):
    """Run deconvolution with logging for a single regularization strength/regularization width
    """
    if regularizer_width == 0:
        deconvolver = deconvolvers.LRDeconvolve(
            impulse_response['x'],
            impulse_response['y'],
            pax_spectra['x'],
            iterations=iterations,
            ground_truth_y=xray_xy['y'],
            X_valid=val_pax_y,
            logging=True
        )
    else:
        deconvolver = deconvolvers.LRFisterDeconvolve(
            impulse_response['x'],
            impulse_response['y'],
            pax_spectra['x'],
            regularization_strength=regularizer_width,
            iterations=iterations,
            ground_truth_y=xray_xy['y'],
            logging=True,
            X_valid=val_pax_y
        )
    deconvolver.fit(np.array(pax_spectra['y']))