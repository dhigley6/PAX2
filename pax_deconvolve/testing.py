import numpy as np
from scipy.signal import convolve

from pax_simulations import simulate_pax
from deconvolution import deconvolvers, assess_convergence
import visualize

# set simulation parameters
LOG10_NUM_ELECTRONS = 5.0  # 10^7 detected electrons (over entire dateset)
RIXS_MODEL = 'schlappa'  # use RIXS model chosen to approximate that in Schlappa's paper
PHOTOEMISSION_MODEL = 'ag'   # use Ag 3d core levels as model photoemission
NUM_SIMULATIONS = 100    # Number of PAX spectra to simulate
ENERGY_LOSS = np.arange(-8, 10, 0.01)  # energy loss values of RIXS to simulate over
REGULARIZATION_STRENGTHS = np.logspace(-3, -1, 10)  # Regularization strengths to try
ITERATIONS = 100    # Number of iterations to run simulations for
CV_FOLD = 3   # Number of folds to use for cross validation

def run():
    # Simulate some PAX data 
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        LOG10_NUM_ELECTRONS,
        RIXS_MODEL,
        PHOTOEMISSION_MODEL,
        NUM_SIMULATIONS,
        ENERGY_LOSS
    )
    deconvolver = deconvolvers.LRFisterGrid(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        REGULARIZATION_STRENGTHS,
        ITERATIONS,
        xray_xy['y'],
        CV_FOLD
    )
    _ = deconvolver.fit(np.array(pax_spectra['y']))
    visualize.plot_result(deconvolver)

def demo_test():
    b2 = 0.15    # Half of photoemission peak broadening
    binding_energies = np.arange(355, 390, 0.02)
    photoemission_spectrum = 5*(b2)/((binding_energies-368.3)**2+(b2)**2)+3*(b2)/((binding_energies-374.0)**2+(b2)**2)
    norm_factor = np.sum(photoemission_spectrum)
    impulse_response = {
        'x': -1*binding_energies,
        'y': np.flipud(photoemission_spectrum)/norm_factor
    }
    photon_energies = np.arange(770, 788, 0.02)
    p1 = 8*np.exp(-((photon_energies-777.75)/0.05)**2)
    p2 = 23*np.exp(-((photon_energies-776.2)/0.2)**2)
    p3 = 26*np.exp(-((photon_energies-775.8)/0.2)**2)
    p4 = 9*np.exp(-((photon_energies-775.1)/0.3)**2)
    p5 = 3*np.exp(-((photon_energies-773.5)/0.5)**2)
    p6 = 3*np.exp(-((photon_energies-772.8)/0.75)**2)
    xray_spectrum = p1+p2+p3+p4+p5+p6
    noiseless_pax_spectrum = convolve(
        xray_spectrum,
        impulse_response['y'],
        mode='valid'
    )
    num_pax_spectra = 5
    single_photon = 5
    counts = 10.0**7   #10^7
    single_photon = num_pax_spectra*np.sum(noiseless_pax_spectrum)/counts
    pax_spectra = []
    for i in range(num_pax_spectra):
        pax_spectrum = np.random.poisson(noiseless_pax_spectrum/single_photon)*single_photon
        pax_spectra.append(pax_spectrum)
    deconvolver = deconvolvers.LRFisterGrid(
        impulse_response['x'],
        impulse_response['y'],
        np.arange(len(noiseless_pax_spectrum)),
        REGULARIZATION_STRENGTHS,
        ITERATIONS,
        xray_spectrum,
        CV_FOLD
    )
    _ = deconvolver.fit(np.array(pax_spectra))
    visualize.plot_result(deconvolver)