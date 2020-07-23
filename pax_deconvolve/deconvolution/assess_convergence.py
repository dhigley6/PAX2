"""Tool for assessing convergence of deconvolution

The intended use of this tool is to run it with the 'run' or 'run_pax_preset' functions
below, then view the results using tensorboard on the log directory.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from pax_deconvolve.deconvolution import deconvolvers
from pax_deconvolve.pax_simulations import simulate_pax

# Set default simulation parameters
DEFAULT_PARAMETERS = {
    "energy_loss": np.arange(-8, 10, 0.005),
    "iterations": 1e2,
    "simulations": 1000,
    "cv_fold": 4,
    "regularizer_widths": np.logspace(-3, -1, 10),
}


def run(
    impulse_response_x, 
    impulse_response_y,
    convolved_x,
    convolved_y, 
    ground_truth_y, 
    regularizer_widths, 
    iterations
):
    """Log deconvolution results as a function of iteration number using tensorboard
    To be used to make sure deconvolutions have been run for sufficient iterations.
    """
    convolved_train_y, convolved_val_y = _split_convolved_data(convolved_y)
    Parallel(n_jobs=-1)(
        delayed(_run_single_deconvolver)(
            impulse_response_x,
            impulse_response_y,
            convolved_x,
            convolved_train_y,
            ground_truth_y,
            regularizer_width,
            iterations,
            convolved_val_y
        )
        for regularizer_width in regularizer_widths
    )
    # below code can be used for debugging in case parallel case doesn't work
    # _ = (run_single_deconvolver(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations, val_pax_y) for regularizer_width in regularizer_widths)


def _split_convolved_data(convolved_y):
    """Split convolved data into a training and validation set
    """
    convolved_train_y, convolved_validation_y = train_test_split(
        convolved_y, test_size=0.3
    )
    convolved_validation_y = np.mean(convolved_validation_y, axis=0)
    return convolved_train_y, convolved_validation_y


def _run_single_deconvolver(
    impulse_response_x,
    impulse_response_y, 
    convolved_x,
    train_convolved_y, 
    ground_truth_y, 
    regularizer_width, 
    iterations, 
    val_pax_y
):
    """Run deconvolution with logging for a single regularization strength/regularization width
    """
    if regularizer_width == 0:
        deconvolver = deconvolvers.LRDeconvolve(
            impulse_response_x,
            impulse_response_y,
            convolved_x,
            iterations=iterations,
            ground_truth_y=ground_truth_y,
            X_valid=val_pax_y,
            logging=True,
        )
    else:
        deconvolver = deconvolvers.LRFisterDeconvolve(
            impulse_response_x,
            impulse_response_y,
            convolved_x,
            regularization_strength=regularizer_width,
            iterations=iterations,
            ground_truth_y=ground_truth_y,
            logging=True,
            X_valid=val_pax_y,
        )
    deconvolver.fit(np.array(train_convolved_y))
