"""Tool for assessing convergence of deconvolution

Approximate convergence procedure:
   - We want the model to overfit the data to the extent possible for each case of
   regularization hyperparameter. This ensures that the regularization parameter controls
   the degree of overfitting to the data and not the number of iterations. Empirically, we
   observed that, for sufficiently small regularization hyperparameters, the validation
   reconstruction error initially decreases with iteration number before reaching a minimum,
   then increasing again, as shown in Fig. ?. As also seen in Fig. ?, the minima of the curves
   that have such a feature are all close in iteration number. For iterations below this minimum, the model
   seems to have underfit the data, while for iterations above the minimum, it seems to have
   overfit the data. To try to have the model always be significantly overfitting the data when
   the regularization hyperparameter is sufficiently small, we made sure to use a number of iterations
   in the reconstruction that is at least ten times as large as the number of iterations where the curve
   with the smallest regularization hyperparameter reaches a minimum.
"""

import numpy as np
from joblib import Parallel, delayed

from pax_deconvolve import LRDeconvolve
from pax_deconvolve.pax_simulations import simulate_pax
from pax_deconvolve import pax_simulation_analysis
DEFAULT_PARAMETERS = pax_simulation_analysis.DEFAULT_PARAMETERS

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
    _, val_pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons-0.33,
        rixs,
        photoemission,
        parameters['simulations'],
        parameters['energy_loss']
    )
    val_pax_y = np.mean(val_pax_spectra['y'], axis=0)
    regularizer_widths = parameters['regularizer_widths']
    regularizer_widths = np.append([0], regularizer_widths)
    run(impulse_response, pax_spectra, xray_xy, regularizer_widths, parameters['iterations'], val_pax_y)

def run(impulse_response, pax_spectra, xray_xy, regularizer_widths, iterations, val_pax_y):
    """Log deconvolution results as a function of iteration number using tensorboard
    To be used to make sure deconvolutions have been run for sufficient iterations.
    """
    Parallel(n_jobs=-1)(delayed(run_single_deconvolver)(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations, val_pax_y) for regularizer_width in regularizer_widths)

def run_single_deconvolver(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations, val_pax_y):
    if regularizer_width == 0:
        deconvolver = LRDeconvolve.LRDeconvolve(
            impulse_response['x'],
            impulse_response['y'],
            pax_spectra['x'],
            iterations=iterations,
            ground_truth_y=xray_xy['y'],
            X_valid=val_pax_y,
            logging=True
        )
    else:
        deconvolver = LRDeconvolve.LRFisterDeconvolve(
            impulse_response['x'],
            impulse_response['y'],
            pax_spectra['x'],
            regularizer_width=regularizer_width,
            iterations=iterations,
            ground_truth_y=xray_xy['y'],
            logging=True,
            X_valid=val_pax_y
        )
    deconvolver.fit(np.array(pax_spectra['y']))