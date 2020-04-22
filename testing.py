"""
Module for testing new ideas
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

import LRDeconvolve
from pax_simulations import simulate_pax
from visualize import plot_photoemission, plot_result, cv_plot
import assess_convergence
import pax_simulation_analysis

PARAMETERS = {
    'energy_loss': np.arange(-8, 10, 0.005),
    'iterations': int(1E1),
    'simulations': int(1E3),
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

from visualize.manuscript_plots import doublet_performance, schlappa_performance
TOTAL_SEPARATION_LIST = doublet_performance.TOTAL_SEPARATION_LIST
TOTAL_LOG10_NUM_ELECTRONS_LIST = doublet_performance.TOTAL_LOG10_NUM_ELECTRONS_LIST

def run_test(log10_num_electrons=10, rixs='schlappa', photoemission='ag'):
    #data = pax_simulation_analysis.run(log10_num_electrons, rixs, photoemission, **PARAMETERS)
    data = pax_simulation_analysis.load(log10_num_electrons, rixs, photoemission)
    plot_photoemission.make_plot(data['deconvolver'])
    cv_plot.make_plot(data['deconvolver'])
    plot_result.make_plot(data['deconvolver'])

def convergence_test2(log10_num_electrons=7, rixs=['doublet', 0.025], photoemission='fermi'):
    parameters = {
        'energy_loss': np.arange(-0.5, 0.5, 0.001),
        'iterations': 1E5,
        'simulations': 1000,
        'cv_fold': 2,
        'regularizer_widths': np.logspace(-4, -2, 10)
    }
    assess_convergence.run_pax_preset(log10_num_electrons, rixs, photoemission, **parameters)

def run_cv_analysis(iterations=1E5):
    #points = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    points = [6.0]
    for log10_num_electrons in points:
        pax_simulation_analysis.run(log10_num_electrons, rixs=['doublet', 0.01], photoemission='fermi', **PARAMETERS)

def run_schlappa_ag_analysis():
    parameters = {
        'energy_loss': np.arange(-8, 10, 0.005),
        'iterations': int(1E2),
        'simulations': 1000,
        'cv_fold': 3,
        'regularizer_widths': np.logspace(-3, -1, 10)
    }
    for log10_num_electrons in schlappa_performance.LOG10_COUNTS_LIST:
        _ = pax_simulation_analysis.run(log10_num_electrons, rixs='schlappa', photoemission='ag', **parameters)

def run_doublet_fermi_analysis():
    parameters = {
        'energy_loss': np.arange(-0.5, 0.5, 0.001),
        'iterations': int(1E5),
        'simulations': 1000,
        'cv_fold': 3,
        'regularizer_widths': np.logspace(-4, -2, 10)
    }
    for separation in TOTAL_SEPARATION_LIST:
        for log10_num_electrons in TOTAL_LOG10_NUM_ELECTRONS_LIST:
            _ = pax_simulation_analysis.run(log10_num_electrons, rixs=['doublet', separation], photoemission='fermi', **parameters)
