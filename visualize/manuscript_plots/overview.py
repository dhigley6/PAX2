#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley
Make an overview plot of how PAX works
"""

import numpy as np
import matplotlib.pyplot as plt

from pax_simulations import model_photoemission, model_rixs, simulate_pax
from visualize import set_plot_params
set_plot_params.init_paper_small()

def make_plot():
    incident_photon_energy = 778
    energy_loss = np.arange(-20, 20, 0.005)
    schlappa_rixs = model_rixs.make_model_rixs('schlappa', energy_loss, incident_photon_energy)
    ag_photoemission = model_photoemission.make_model_photoemission('ag', schlappa_rixs, schlappa_rixs['x'][1]-schlappa_rixs['x'][0])
    _, pax_spectra = simulate_pax.simulate(schlappa_rixs, ag_photoemission, 1E10, 1)
    pax_spectrum = {
        'x': pax_spectra['x'],
        'y': pax_spectra['y'][0]}
    f, axs = plt.subplots(3, 1, figsize=(2.37, 4))
    axs[0].plot(schlappa_rixs['x'], schlappa_rixs['y'], color='k')
    axs[0].set_xlim((770, 780))
    axs[1].plot(ag_photoemission['x'], ag_photoemission['y'], color='k')
    axs[1].set_xlim((365, 380))
    axs[2].plot(pax_spectrum['x'], pax_spectrum['y'], color='k')
    axs[2].set_xlim((395, 412))
    format_plot(f, axs)
    #plt.savefig('../plots/2019_09_23_pax_overview.eps', dpi=600)
    
def format_plot(f, axs):
    axs[0].yaxis.set_ticklabels([])
    axs[0].yaxis.set_ticks([])
    axs[1].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticks([])
    axs[2].yaxis.set_ticks([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[1].set_xlabel('Binding Energy (eV)')
    axs[2].set_xlabel('Kinetic Energy (eV)')
    axs[1].set_ylabel('Intensity')
    #axs[0].set_xlim((776.5, 778.5))
    #axs[1].set_xlim((365, 380))
    #axs[2].set_xlim((400, 412))
    axs[0].text(0.9, 0.8, 'A', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.8, 'B', transform=axs[1].transAxes)
    axs[2].text(0.9, 0.8, 'C', transform=axs[2].transAxes)
    axs[0].text(0.3, 0.95, 'Desired RIXS\nr(E)', transform=axs[0].transAxes,
                horizontalalignment='center', verticalalignment='top', 
                fontsize=8)
    axs[1].text(0.55, 0.95, 'Photoemission\np(E)', transform=axs[1].transAxes,
                horizontalalignment='center', verticalalignment='top', fontsize=8)
    axs[2].text(0.35, 0.95, 
       ''.join(['Measured PAX\n', r'$m(E) = r(E)\ast p(-E)$']), 
       transform=axs[2].transAxes,
       horizontalalignment='center', verticalalignment='top', fontsize=8)
    #axs[1].text(0.9, 0.7, ')
    plt.tight_layout(w_pad=0, h_pad=0)