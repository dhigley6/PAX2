#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:03:51 2019
@author: dhigley
Calculate simulated PAX spectra with Poisson statistics.
"""

import numpy as np
from scipy.signal import convolve

from pax_simulations import model_rixs
from pax_simulations import model_photoemission

def simulate_from_presets(total_log10_num_electrons, rixs, photoemission, num_simulations, energy_spacing, bg_height=0.05, bg_width=100):
    """Simulate PAX spectra using preset options for RIXS and photoemission spectra
    """
    total_counts = 10**total_log10_num_electrons
    xray_xy = model_rixs.make_model_rixs(rixs, energy_spacing)
    photoemission_xy = model_photoemission.make_model_photoemission(photoemission, xray_xy, energy_spacing)
    impulse_response, pax_spectra = simulate(
        xray_xy,
        photoemission_xy,
        total_counts,
        num_simulations,
        bg_height,
        bg_width
    )
    return impulse_response, pax_spectra, xray_xy

def simulate(xray_spectrum, photoemission_spectrum, counts, bg_height=0.2, bg_width=100, num_simulations=1):
    """Simulate PAX spectra 
    """
    impulse_response = calculate_pax_impulse_response(photoemission_spectrum)
    noiseless_pax_spectrum = convolve(xray_spectrum['y'],
                                         impulse_response['y'],
                                         mode='valid')
    single_photon = np.sum(noiseless_pax_spectrum/counts)
    pax_y_list = [_apply_poisson_noise(noiseless_pax_spectrum, single_photon) for _ in range(num_simulations)]
    pax_spectra = {
            'x': _calculate_pax_kinetic_energy(
                    xray_spectrum,
                    photoemission_spectrum),
            'y': pax_y_list,
            'x_min': xray_spectrum['x_min']-photoemission_spectrum['x_max'],
            'x_max': xray_spectrum['x_max']-photoemission_spectrum['x_min']}
    return impulse_response, pax_spectra
    
def calculate_pax_impulse_response(photoemission_spectrum):
    """Normalize and flip photoemission to obtain PAX impulse response.
    """
    impulse_response = {'x': -1*photoemission_spectrum['x'],
                        'y': np.flipud(photoemission_spectrum['y'])}
    norm_factor = np.sum(impulse_response['y'])
    impulse_response['y'] = impulse_response['y']/norm_factor
    if 'bg' in photoemission_spectrum.keys():
        impulse_response_bg = np.flipud(photoemission_spectrum['bg'])
        impulse_response_bg = impulse_response_bg/norm_factor
        impulse_response.update({'bg': impulse_response_bg})
    return impulse_response

def _apply_poisson_noise(data, single_photon=1.0):
    """Apply Poisson noise to input data
    
    single_photon is the number of counts that corresponds to a single
    detected photon.
    """
    #data_clipped_below_zero = np.clip(data, 1E-6, None)
    output = np.random.poisson(data/single_photon)*single_photon
    return output

def _calculate_pax_kinetic_energy(xray_spectrum, photoemission_psf):
    photon_energy_in = xray_spectrum['x']
    average_binding_energy = np.mean(photoemission_psf['x'])
    kinetic_energy = photon_energy_in-average_binding_energy
    return kinetic_energy