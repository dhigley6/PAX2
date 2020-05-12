#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:41:16 2019
@author: dhigley
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

from pax_deconvolve import pax_simulation_pipeline
from pax_deconvolve.visualize import set_plot_params
set_plot_params.init_paper_small()

TOTAL_SEPARATION_LIST = [0.025, 0.05, 0.1, 0.15, 0.2]
TOTAL_LOG10_NUM_ELECTRONS_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
SPECTRA_SEPARATION_LIST = [0.05]
SPECTRA_LOG10_NUM_ELECTRONS_LIST = [3.0, 5.0, 7.0]

FIGURES_DIR = 'figures'
def make_figure():
    """Figure assessing capability of PAX to resolve a doublet
    axis 0: Example target doublet spectrum, deconvolved spectra vs number of
        detected electrons (one not resolved, one just resolved, one well
        resolved)
    axis 1: Minimum resolved separation vs. detected number of electrons
    """
    spectra_data = _get_spectra_data()
    total_data = _get_total_data()
    _, axs = plt.subplots(2, 1, figsize=(3.37, 4.5))
    _make_spectra_plot(axs[0], spectra_data[0])
    _min_resolved_plot(axs[1], total_data, TOTAL_SEPARATION_LIST, TOTAL_LOG10_NUM_ELECTRONS_LIST)
    #_min_rmse_plot(axs[1], total_data, TOTAL_SEPARATION_LIST, TOTAL_LOG10_NUM_ELECTRONS_LIST)
    _format_figure(axs)
    file_name = f'{FIGURES_DIR}/pax_performance2.eps'
    plt.savefig(file_name, dpi=600)

def _make_spectra_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        data = data['deconvolver']
        offset = 2.0-ind*1.0
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
            data = pax_simulation_pipeline.load(log10_num_electrons, rixs=['doublet', separation], photoemission='fermi')
            data_list.append(data)
        data_list_list.append(data_list)
    return data_list_list
    
def _format_figure(axs):
    axs[0].set_xlim((-0.1, 0.15))
    axs[0].set_ylim((-0.1, 3.4))
    axs[0].invert_xaxis()
    axs[0].set_xlabel('Energy Loss (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Doublet Separation (eV)')
    legend_elements = [Line2D([0], [0], color='k', linestyle='--', label='Ground Truth'),
                       Line2D([0], [0], color='r', label='Deconvolved')]
    axs[0].legend(handles=legend_elements, loc='upper left', frameon=False, ncol=2)
    axs[1].set_ylabel('Minimum Counts for\nPeak Intensity More than\n5x Gap Strength')
    for ind, log10_num_electrons in enumerate(SPECTRA_LOG10_NUM_ELECTRONS_LIST):
        axs[0].text(0.125, 2+0.25-ind, '10$^'+str(int(log10_num_electrons))+'$')
    plt.tight_layout()
    axs[0].text(0.9, 0.1, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.9, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)

def _min_rmse_plot(ax, data_list_list, separations, log10_num_electrons):
    min_rmse_list = []
    for data_list, separation in zip(data_list_list, separations):
        norm_rmse_threshold_list = []
        for ind, data in enumerate(data_list):
            data = data['deconvolver']
            print(ind, separation)
            in_range = (data.deconvolved_x >= (778-2*separation)) & (data.deconvolved_x < 778)
            deconvolved_mse = mean_squared_error(data.deconvolved_y_[in_range], data.ground_truth_y[in_range])
            rmse = np.sqrt(deconvolved_mse)
            norm_rmse = rmse/np.amax(data.ground_truth_y[in_range])
            norm_rmse_threshold = norm_rmse < 0.1
            norm_rmse_threshold_list.append(norm_rmse_threshold)
        if np.sum(norm_rmse_threshold_list) > 0:
            min_threshold_ind = np.arange(len(norm_rmse_threshold_list))[norm_rmse_threshold_list][0]
            min_threshold_num_electrons = 10**log10_num_electrons[min_threshold_ind]
        else:
            min_threshold_num_electrons = np.nan
        print(min_threshold_num_electrons)
        min_rmse_list.append(min_threshold_num_electrons)
    ax.semilogy(separations, min_rmse_list, marker='o', color='k', linestyle='None')

    
def _min_resolved_plot(ax, data_list_list, separations, log10_num_electrons):
    min_resolved_list = []
    for data_list, separation in zip(data_list_list, separations):
        print(separation)
        resolved_list = []
        for data in data_list:
            data = data['deconvolver']
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
    spread = np.abs(pos2-pos1)
    middle = np.mean([pos2, pos1])
    int_range_1 = (spectrum_x > pos1-spread*0.1) & (spectrum_x < pos1+spread*0.1)
    int_range_2 = (spectrum_x > pos2-spread*0.1) & (spectrum_x < pos2+spread*0.1)
    int_range_gap = (spectrum_x > middle-spread*0.1) & (spectrum_x < middle+spread*0.1)
    int_1 = np.mean(spectrum_y[int_range_1])
    int_2 = np.mean(spectrum_y[int_range_2])
    int_gap = np.mean(spectrum_y[int_range_gap])
    int_peaks = np.mean([int_1, int_2])
    sum_peaks = np.sum(spectrum_y[int_range_1])+np.sum(spectrum_y[int_range_2])
    sum_total = np.sum(spectrum_y)
    resolved = (int_1 > 5*int_gap) & (int_2 > 5*int_gap) & (sum_peaks > 0.7*sum_total)
    print(sum_peaks/sum_total)
    print(resolved)
    print(spread)
    if ((spread > 0.04) & (spread < 0.06)):
        plt.figure()
        plt.plot(spectrum_x, int_range_1)
        plt.plot(spectrum_x, int_range_2)
        plt.plot(spectrum_x, int_range_gap)
        plt.plot(spectrum_x, spectrum_y/np.amax(spectrum_y))
        plt.title(str(resolved))
    return resolved