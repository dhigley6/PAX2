import numpy as np

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
