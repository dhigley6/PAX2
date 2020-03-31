#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:24:04 2019
@author: dhigley
"""

import numpy as np
import matplotlib.pyplot as plt

def make_plot(deconvolved, sim):
    """Summarize deconvolution results
    deconvolved: fitted deconvolver
    sim: Simulated PAX data set
    """
    _, axs = plt.subplots(3, 2, figsize=(7, 7))
    _make_deconvolved_plot(axs[0, 0], deconvolved, sim)
    _make_reconvolved_plot(axs[1, 0], deconvolved, sim)
    #_make_deconvolved_spectra_plot(axs[0, 0], data_list)
    #_make_reconvolved_spectra_plot(axs[1, 0], data_list)
    #_deconvolved_plot(axs[0, 1], data_list)
    #_reconvolved_plot(axs[1, 1], data_list)
    #_difference_from_previous_plot(axs[2, 1], data_list)
    _format_plot(axs)

def _make_deconvolved_plot(ax, deconvolved, sim):
    ax.plot(sim['xray_xy']['x'], sim['xray_xy']['y'], label='Ground Truth')
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_, label='Deconvolved')

def _make_reconvolved_plot(ax, deconvolved, sim):
    ax.plot(sim['mean_pax_xy']['x'], sim['mean_pax_xy']['y'], label='PAX')
    ax.plot(deconvolved.reconvolved_x, deconvolved.reconvolved_y_, label='Reconvolved')

def _deconvolved_plot(ax, data_list):
    deconvolved_mse_list = [postprocess.deconvolved_mse_history(data) for data in data_list]
    deconvolved_mse_mean = np.mean(deconvolved_mse_list, axis=0)
    deconvolved_mse_std = np.std(deconvolved_mse_list, axis=0)
    ax.errorbar(data_list[0]['history']['iterations'], deconvolved_mse_mean, deconvolved_mse_std)

def _reconvolved_plot(ax, data_list):
    reconvolved_mse_list = [postprocess.reconvolved_mse_history(data) for data in data_list]
    reconvolved_mse_mean = np.mean(reconvolved_mse_list, axis=0)
    reconvolved_mse_std = np.std(reconvolved_mse_list, axis=0)
    ax.errorbar(data_list[0]['history']['iterations'], reconvolved_mse_mean, reconvolved_mse_std)

def _difference_from_previous_plot(ax, data_list):
    difference_list = [data['history']['deconvolved_difference_from_previous'] for data in data_list]
    difference_mean = np.mean(difference_list, axis=0)
    difference_std = np.std(difference_list, axis=0)
    ax.errorbar(data_list[0]['history']['iterations'], difference_mean, difference_std)
    
def _format_plot(axs):
    axs[0, 0].legend(loc='best')
    axs[1, 0].legend(loc='best')
    axs[0, 0].set_xlabel('Photon Energy (eV)')
    axs[0, 0].set_ylabel('Intensity (a.u.)')
    axs[1, 0].set_xlabel('Electron Energy (eV)')
    axs[1, 0].set_ylabel('Intensity (a.u.)')
    plt.tight_layout()
    #axs[0, 1].set_yscale('log')
    #axs[1, 1].set_yscale('log')
    #axs[2, 1].set_yscale('log')
    #axs[0, 0].legend(loc='best')
    #axs[1, 0].legend(loc='best')
    #axs[0, 1].set_xlabel('Iterations')
    #axs[0, 1].set_ylabel('RMS Deconvolved\nError')
    #axs[1, 1].set_xlabel('Iterations')
    #axs[1, 1].set_ylabel('RMS Reconvolved\nError')
    #axs[2, 1].set_xlabel('Iterations')
    #axs[2, 1].set_ylabel('RMS Deconvolved\nDifference from Previous')

