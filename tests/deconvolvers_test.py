"""Tests for deconvolvers.py
"""

import numpy as np
import pytest
from scipy.signal import convolve
from sklearn.metrics import mean_squared_error

from pax_deconvolve.deconvolution import deconvolvers


def test_normalized_gausian_is_normalized():
    x = np.arange(100)
    mu = np.mean(x)
    sigma = 3
    norm_gauss = deconvolvers._normalized_gaussian(x, mu, sigma)
    assert np.sum(norm_gauss) == 1


def test_get_deconvolved_x():
    spacing = 0.1
    ground_truth_x = np.arange(-1, 10, spacing) + 5
    impulse_response_x = np.arange(0, 20, spacing) + 2
    first_point = ground_truth_x[0] + impulse_response_x[-1]
    convolved_length = len(impulse_response_x) - len(ground_truth_x) + 1
    convolved_x = np.arange(
        first_point, first_point + convolved_length * spacing, spacing
    )
    re_deconvolved_x = deconvolvers._get_deconvolved_x(convolved_x, impulse_response_x)
    assert np.allclose(ground_truth_x, re_deconvolved_x)


@pytest.fixture
def dummy_convolved_data():
    """Return dummy convolved data"""
    spacing = 0.01
    ground_truth_x = np.arange(765, 783, spacing)
    p1 = 8 * np.exp(-(((-777.75) / 0.05) ** 2))
    p2 = 23 * np.exp(-(((ground_truth_x - 776.2) / 0.2) ** 2))
    p3 = 26 * np.exp(-(((ground_truth_x - 775.8) / 0.2) ** 2))
    p4 = 9 * np.exp(-(((ground_truth_x - 775.1) / 0.3) ** 2))
    p5 = 3 * np.exp(-(((ground_truth_x - 773.5) / 0.5) ** 2))
    p6 = 3 * np.exp(-(((ground_truth_x - 772.8) / 0.75) ** 2))
    ground_truth_y = p1 + p2 + p3 + p4 + p5 + p6
    impulse_response_x = np.arange(355, 390, spacing)
    b2 = 0.15
    impulse_response_y = 5 * (b2) / (
        (impulse_response_x - 368.3) ** 2 + (b2) ** 2
    ) + 3 * (b2) / ((impulse_response_x - 374.0) ** 2 + (b2) ** 2)
    impulse_response_y = impulse_response_y / np.sum(impulse_response_y)
    noiseless_convolved_y = convolve(ground_truth_y, impulse_response_y, mode="valid")
    convolved_x = np.linspace(
        0, spacing * (len(noiseless_convolved_y) - 1), len(noiseless_convolved_y)
    )
    num_spectra = 500
    log10_num_electrons = 5.0
    counts = 10.0 ** log10_num_electrons
    single_photon = num_spectra * np.sum(noiseless_convolved_y) / counts
    convolved_y = []
    for _ in range(num_spectra):
        current = (
            np.random.poisson(noiseless_convolved_y / single_photon) * single_photon
        )
        convolved_y.append(current)
    convolved_y = np.array(convolved_y)
    dummy = {
        "ground_truth_x": ground_truth_x,
        "ground_truth_y": ground_truth_y,
        "impulse_response_x": impulse_response_x,
        "impulse_response_y": impulse_response_y,
        "convolved_x": convolved_x,
        "convolved_y": convolved_y,
    }
    return dummy


def test_single_LR_iteration(dummy_convolved_data):
    """Confirm that a single Lucy-Richardson iteration works as expected"""
    initial_guess = np.ones_like(dummy_convolved_data["ground_truth_y"])
    initial_guess = (
        initial_guess
        * np.sum(dummy_convolved_data["ground_truth_y"])
        / np.sum(initial_guess)
    )
    measured_y = np.mean(dummy_convolved_data["convolved_y"], axis=0)
    result = deconvolvers._LR_iteration(
        measured_y,
        dummy_convolved_data["impulse_response_y"],
        initial_guess,
        np.flip(dummy_convolved_data["impulse_response_y"]),
        np.ones_like(measured_y),
    )
    impulse_response_y_reversed = np.flip(dummy_convolved_data["impulse_response_y"])
    ones_vec = np.ones_like(np.mean(dummy_convolved_data["convolved_y"], axis=0))
    blurred = convolve(
        dummy_convolved_data["impulse_response_y"], initial_guess, mode="valid"
    )
    correction_factor = np.mean(dummy_convolved_data["convolved_y"], axis=0) / blurred
    gradient_term1 = convolve(impulse_response_y_reversed, ones_vec, mode="valid")
    gradient_term2 = -1 * convolve(
        impulse_response_y_reversed, correction_factor, mode="valid"
    )
    gradient = gradient_term1 + gradient_term2
    expected = initial_guess * (1 - gradient)
    assert np.allclose(expected, result)


def test_reasonable_deconvolution_result(dummy_convolved_data):
    """Confirm that deconvolution works to degree expected"""
    np.random.seed(0)
    deconvolver = deconvolvers.LRFisterGrid(
        dummy_convolved_data["impulse_response_x"],
        dummy_convolved_data["impulse_response_y"],
        dummy_convolved_data["convolved_x"],
        regularization_strengths=np.logspace(-3, -1, 10),
        iterations=100,
        ground_truth_y=dummy_convolved_data["ground_truth_y"],
        cv_folds=3,
    )
    _ = deconvolver.fit(dummy_convolved_data["convolved_y"])
    mse = mean_squared_error(deconvolver.ground_truth_y, deconvolver.deconvolved_y_)
    assert mse < 1
