#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley
Calculate simulated PAX spectra with Poisson statistics, and a bg of the
photoemission spectrum that was used to calculate the PAX spectra. Also,

"""

import model_rixs
import model_photoemission
import simulate_pax

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
    results = simulate_pax.simulate_set_from_presets(total_log10_num_electrons, rixs, photoemission, num_simulations, energy_spacing)



def subtract_bg_from_pax():
    pass