"""
Testing performance of PAX with model Fermi edge photoemission converter
"""

import numpy as np

from pax_deconvolve.pax_simulations import run_analyze_save_load

# Simulation parameters:
PARAMETERS = {
    'energy_spacing': 0.005,
    'iterations': 1E4,
    'simulations': 1E3,
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}


def run_fermi_georgi_simulations():
    for num_electrons in np.logspace(3, 7, 5):
        run_analyze_save_load.run(np.log10(num_electrons), photoemission='fermi', rixs='georgi', **PARAMETERS)

def run_fermi_schlappa_simulations():
    for num_electrons in np.logspace(3, 7, 5):
        run_analyze_save_load.run(np.log10(num_electrons), photoemission='fermi', rixs='schlappa', **PARAMETERS)

def run_ag_georgi_simulations():
    for num_electrons in np.logspace(3, 7, 5):
        run_analyze_save_load.run(np.log10(num_electrons), photoemission='ag', rixs='georgi', **PARAMETERS)

def run_ag_schlappa_simulations():
    for num_electrons in np.logspace(3, 7, 5):
        run_analyze_save_load.run(np.log10(num_electrons), photoemission='ag', rixs='schlappa', **PARAMETERS)