"""Tool for assessing convergence of deconvolution

The intended use of this tool is to run it with the 'run' or 'run_pax_preset' functions
below, then view the results using tensorboard on the log directory.
"""

from typing import List, Optional, Tuple
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from pax_deconvolve.deconvolution import deconvolvers


def run(
    impulse_response_x: np.ndarray,
    impulse_response_y: np.ndarray,
    convolved_x: np.ndarray,
    convolved_y: np.ndarray,
    ground_truth_y: np.ndarray,
    regularizer_widths: List[float],
    iterations: int,
):
    """Log deconvolution results as a function of iteration number using tensorboard

    To be used to make sure deconvolutions have been run for sufficient iterations.

    Parameters:
    -----------
    impulse_response_x : (N,) array_like
        One-dimensional x-values (locations) of impulse response
    impulse_response_y: (N,) array_like
        One-dimensional y-values (intensities) of impulse response
    convolved_x: (M,) array_like
        One-dimensional x-values (locations) of convolved data
    convolved_y: (M, A) array_like
        Two-dimensional y-values (intensities) of convolved data
        Each row is a separate measurement/simulation and each column is a separate location
    ground_truth_y: (N-M+1,) array_like
        One-dimensional y-values (intensities) of ground truth deconvolved data
    regularizer_widths: list of floats
        regularization strengths to test
    iterations: int
        Number of iterations to do
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
            convolved_val_y,
        )
        for regularizer_width in regularizer_widths
    )
    # below code can be used for debugging in case parallel case doesn't work
    # _ = (run_single_deconvolver(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations, val_pax_y) for regularizer_width in regularizer_widths)


def _split_convolved_data(convolved_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split convolved data into a training and validation set

    Parameters:
    -----------
    convolved_y: (M, A) array_like
        Two-dimensional y-values (intensities) of convolved data

    Returns:
    --------
    convolved_train_y: (M, B) array_like
        Two-dimensional y-values (intensities) of convolved data in training set
    convolved_validation_y: (M,) array_like
        One-dimenional y-values (intensities) of convolved data in validation set
    """
    convolved_train_y, convolved_validation_y = train_test_split(
        convolved_y, test_size=0.3
    )
    convolved_validation_y = np.mean(convolved_validation_y, axis=0)
    return convolved_train_y, convolved_validation_y


def _run_single_deconvolver(
    impulse_response_x: np.ndarray,
    impulse_response_y: np.ndarray,
    convolved_x: np.ndarray,
    train_convolved_y: np.ndarray,
    ground_truth_y: np.ndarray,
    regularizer_width: float,
    iterations: int,
    val_convolved_y: np.ndarray,
):
    """Run deconvolution with logging for a single regularization strength/regularization width

    Parameters:
    -----------
    impulse_response_x: (N,) array_like
        One-dimensional x-values (locations) of impulse response
    impulse_response_y: (N,) array_like
        One-dimensional y-values (intensities) of impulse response
    convolved_x: (M,) array_like
        One-dimensional x-values (locations) of impulse response
    train_convolved_y: (M, A) array_like
        Two-dimensional y-values (intensities) of training part of convolved data
    ground_truth_y: (N-M+1,) array_like
        One-dimensional y-values (intensities) of ground truth of deconvolution
    regularizer_width: float
        Regularization strength to use
    iterations: int
        Number of iterations to run
    val_convolved_y: (M,) array_like
        One-dimensional y-values (intensities) of validation part of convolved data
    """
    if regularizer_width == 0:
        deconvolver = deconvolvers.LRDeconvolve(
            impulse_response_x,
            impulse_response_y,
            convolved_x,
            iterations=iterations,
            ground_truth_y=ground_truth_y,
            X_valid=val_convolved_y,
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
            X_valid=val_convolved_y,
        )
    deconvolver.fit(np.array(train_convolved_y))
