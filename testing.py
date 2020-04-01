"""
Module for testing new ideas
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV

from pax_simulations import simulate_pax, model_photoemission, model_rixs, run_analyze_save_load
from visualize import plot_result, summary, cv_plot
import LRDeconvolve

DEFAULT_PARAMETERS = {
    'energy_spacing': 0.005,
    'iterations': 1E2,
    'simulations': 10,
    'cv_fold': 2,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def run_test3(log10_num_electrons, rixs='schlappa', photoemission='ag'):
    #run_analyze_save_load.run(log10_num_electrons, rixs, photoemission)
    saved = run_analyze_save_load.load(log10_num_electrons, rixs, photoemission)
    plot_result.make_plot(saved['deconvolver_gs'].best_estimator_, saved['sim'])
    summary.make_plot(saved['deconvolver_gs'].best_estimator_, saved['sim'])
    cv_plot.make_plot(saved['deconvolver_gs'])