"""Tests for deconvolvers.py
"""

import numpy as np
import pytest

from pax_deconvolve.deconvolution import deconvolvers

def test_normalized_gausian_is_normalized():
    x = np.arange(100)
    mu = np.mean(x)
    sigma = 3
    norm_gauss = deconvolvers._normalized_gaussian(x, mu, sigma)
    assert np.sum(norm_gauss) == 1

def test_get_deconvolved_x():
    spacing = 0.1
    ground_truth_x = np.arange(-1, 10, spacing)+5
    impulse_response_x = np.arange(0, 20, spacing)+2
    first_point = ground_truth_x[0]+impulse_response_x[-1]
    convolved_length = len(impulse_response_x)-len(ground_truth_x)+1
    convolved_x = np.arange(first_point, first_point+convolved_length*spacing, spacing)
    re_deconvolved_x = deconvolvers._get_deconvolved_x(convolved_x, impulse_response_x)
    assert np.allclose(ground_truth_x, re_deconvolved_x)


'''
@pytest.fixture
def get_dummy_LR():
    impulse_response_x = np.linspace(0, 100, 1000)
    impulse_response_y = np.exp((impulse_response_x-50)**2)
    convolved_x = np.linspace(0, 50, 500)
    LR = deconvolvers.LRDeconvolve(
        impulse_response_x,
        impulse_response_y,
        convolved_x
    )
    return LR

def test_LRDeconvolve_fit(get_dummy_LR):
    dummy_convolved_y = np.linspace()
    get_dummy_LR.fit()
    

@pytest.fixture
def get_dummy_data():
    """Get dummy data for deconvolution"""
    pass

@pytest.fixture
def get_dummy_irf():
    """Get dummy impulse response function"""
    b2 = 0.15    # Half of photoemission peak broadening
    binding_energies = np.arange(355, 390, 0.01)
    photoemission_spectrum = 5*(b2)/((binding_energies-368.3)**2+(b2)**2)+3*(b2)/((binding_energies-374.0)**2+(b2)**2)
    photoemission_spectrum = photoemission_spectrum/np.sum(photoemission_spectrum)
    impulse_response = {
        'x': -1*binding_energies,
        'y': np.flipud(photoemission_spectrum)
    }
    return impulse_response

@pytest.fixture
def get_dummy_ground_truth():
    """Get dummy ground truth"""
    pass
'''