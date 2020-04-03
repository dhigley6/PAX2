#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:03:51 2019
@author: dhigley
Calculate simulated PAX spectra with Poisson statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from pax_simulations import model_rixs
from pax_simulations import model_photoemission

def simulate_set_from_presets(total_log10_num_electrons, rixs, photoemission, num_simulations, energy_spacing, bg_height=0.05, bg_width=100):
    total_counts = 10**total_log10_num_electrons
    xray_xy = model_rixs.make_model_rixs(rixs, energy_spacing)
    photoemission_xy = model_photoemission.make_model_photoemission(photoemission, xray_xy, energy_spacing)
    impulse_response, pax_x, pax_y_list, mean_pax_xy = simulate_pax_set(
        xray_xy,
        photoemission_xy,
        total_counts,
        num_simulations,
        bg_height,
        bg_width
    )
    to_return = {
        'impulse_response': impulse_response,
        'pax_x': pax_x,
        'pax_y_list': pax_y_list,
        'mean_pax_xy': mean_pax_xy,
        'xray_xy': xray_xy
    }
    if 'bg' in impulse_response.keys():
        to_return.update({'impulse_response_bg': impulse_response['bg']})
    return to_return

def simulate(xray_spectrum, photoemission_spectrum, counts, bg_height=0.2, bg_width=100):
    """Simulate PAX spectrum
    """
    impulse_response = calculate_pax_impulse_response(photoemission_spectrum)
    noiseless_pax_spectrum = convolve(xray_spectrum['y'],
                                         impulse_response['y'],
                                         mode='valid')
    single_photon = np.sum(noiseless_pax_spectrum/counts)
    pax_spectrum_y = _apply_poisson_noise(noiseless_pax_spectrum, single_photon)
    pax_spectrum = {
            'x': _calculate_pax_kinetic_energy(
                    xray_spectrum,
                    photoemission_spectrum),
            'y': pax_spectrum_y,
            'x_min': xray_spectrum['x_min']-photoemission_spectrum['x_max'],
            'x_max': xray_spectrum['x_max']-photoemission_spectrum['x_min']}
    return impulse_response, pax_spectrum

def simulate_pax_set(rixs, photoemission, total_counts, num_simulations, bg_height, bg_width):
    pax_y_list = []
    for _ in np.arange(num_simulations):
        impulse_response, pax = simulate(rixs, photoemission, round(total_counts/num_simulations), bg_height, bg_width)
        pax_y_list.append(pax['y'])
    pax_x = pax['x']
    pax_mean = np.mean(pax_y_list, axis=0)
    mean_pax_xy = {'x': pax_x,
                   'y': pax_mean}
    return impulse_response, pax_x, pax_y_list, mean_pax_xy
    
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