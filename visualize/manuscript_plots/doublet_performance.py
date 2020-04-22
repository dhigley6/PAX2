#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:41:16 2019
@author: dhigley
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import pax_simulation_analysis

TOTAL_SEPARATION_LIST = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25]
TOTAL_LOG10_NUM_ELECTRONS_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
SPECTRA_SEPARATION_LIST = [0.15]
SPECTRA_LOG10_NUM_ELECTRONS_LIST = [2.0, 3.0, 4.0, 5.0]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../figures')

def make_figure():
    """Figure assessing capability of PAX to resolve a doublet
    axis 0: Example target doublet spectrum, deconvolved spectra vs number of
        detected electrons (one not resolved, one just resolved, one well
        resolved)
    axis 1: Minimum resolved separation vs. detected number of electrons
    """
    data = _get_total_data()
    spectra_data = _get_spectra_data()
    total_data = _get_total_data()
    _, axs = plt.subplots(2, 1, figsize=(3.37, 4.5))
    _make_spectra_plot(axs[0], spectra_data[0])
    _min_resolved_plot(axs[1], total_data, TOTAL_SEPARATION_LIST, TOTAL_LOG10_NUM_ELECTRONS_LIST)
    _format_figure(axs)
    #file_name = f'{FIGURES_DIR}/pax_performance2.eps'
    #plt.savefig(file_name, dpi=600)

def _make_spectra_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        data = data['deconvolver']
        offset = ind*1.0
        norm = 1.1*np.amax(data.ground_truth_y)
        energy_loss = np.flipud(data.deconvolved_x-778)
        ax.plot(energy_loss, 
                offset+data.deconvolved_y_/norm, 'r',
                label='Deconvolved')
        ax.plot(energy_loss,
                offset+data.ground_truth_y/norm, 'k--',
                label='Ground Truth')

def _get_spectra_data():
    spectra_data = _get_specified_data(SPECTRA_SEPARATION_LIST, SPECTRA_LOG10_NUM_ELECTRONS_LIST)
    return spectra_data

def _get_total_data():
    total_data = _get_specified_data(TOTAL_SEPARATION_LIST, TOTAL_LOG10_NUM_ELECTRONS_LIST)
    return total_data

def _get_specified_data(separation_list, log10_num_electrons_list):
    data_list_list = []
    for separation in separation_list:
        data_list = []
        for log10_num_electrons in log10_num_electrons_list:
            data = pax_simulation_analysis.load(log10_num_electrons, rixs=['doublet', separation], photoemission='fermi')
            data_list.append(data)
        data_list_list.append(data_list)
    return data_list_list
    
def _format_figure(axs):
    axs[0].set_xlim((-0.3, 0.45))
    axs[0].invert_xaxis()
    axs[0].set_xlabel('Energy Loss (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Doublet Separation (eV)')
    axs[1].set_ylabel('Minimum Counts\nTo Resolve')
    for ind, log10_num_electrons in enumerate(SPECTRA_LOG10_NUM_ELECTRONS_LIST):
        axs[0].text(0.4, 0.25+ind, '10$^'+str(int(log10_num_electrons))+'$')
    plt.tight_layout()
    axs[0].text(0.9, 0.9, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.9, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    
def _min_resolved_plot(ax, data_list_list, separations, log10_num_electrons):
    min_resolved_list = []
    for data_list, separation in zip(data_list_list, separations):
        resolved_list = []
        for ind, data in enumerate(data_list):
            data = data['deconvolver']
            print(ind, separation)
            resolved = is_doublet_resolved(
                    data.deconvolved_x,
                    data.deconvolved_y_,
                    778-separation,
                    778)
            resolved_list.append(resolved)
        if np.sum(resolved_list) > 0:
            min_resolved_ind = np.arange(len(resolved_list))[resolved_list][0]
            min_resolved_num_electrons = 10**log10_num_electrons[min_resolved_ind]
        else:
            min_resolved_num_electrons = np.nan
        print(min_resolved_num_electrons)
        min_resolved_list.append(min_resolved_num_electrons)
    ax.semilogy(separations, min_resolved_list, marker='o')
    
def is_doublet_resolved(spectrum_x, spectrum_y, pos1, pos2):
    ind1 = np.argmin(np.abs(spectrum_x-pos1))
    ind2 = np.argmin(np.abs(spectrum_x-pos2))
    first_peak_value = spectrum_y[ind1]
    second_peak_value = spectrum_y[ind2]
    min_peak_value = min(first_peak_value, second_peak_value)
    between_peaks = spectrum_y[ind1:ind2]
    if np.amin(between_peaks) < 0.7*min_peak_value:
        return True
    else:
        return False