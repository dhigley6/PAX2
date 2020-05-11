"""
Make plot of simulated deconvolved results with Schlappa RIXS
as a function of the number of detected electrons
"""

import numpy as np 
import matplotlib.pyplot as plt

from pax_deconvolve.pax_simulations import run_analyze_save_load

NUM_ELECTRONS = np.logspace(3, 7, 5)

def make_figure(photoemission='fermi', rixs='schlappa'):
    data = []
    for num_electrons in NUM_ELECTRONS:
        data.append(run_analyze_save_load.load(np.log10(num_electrons), rixs, photoemission))
    f = plt.figure(figsize=(3.37, 5.5))
    grid = plt.GridSpec(4, 2)
    ax_spectra = f.add_subplot(grid[:2, :])
    ax_deconvolved_mse = f.add_subplot(grid[2, :])
    ax_fwhm = f.add_subplot(grid[3, :], sharex=ax_deconvolved_mse)
    _spectra_plot(ax_spectra, data)
    _deconvolved_mse_plot(ax_deconvolved_mse, NUM_ELECTRONS, data)
    _fwhm_plot(ax_fwhm, NUM_ELECTRONS, data)
    axs = [ax_spectra, ax_deconvolved_mse, ax_fwhm]
    _format_figure(axs, data) 

def _spectra_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        deconvolved = data['deconvolver']
        offset = ind*1.0
        norm = 1.1*np.amax(deconvolved.ground_truth_y)
        ax.plot(deconvolved.deconvolved_x, offset+deconvolved.deconvolved_y_/norm, 'r', label='Deconvolved')
        ax.plot(deconvolved.deconvolved_x, offset+deconvolved.ground_truth_y/norm, 'k--', label='Ground Truth')

def _deconvolved_mse_plot(ax, num_electrons, data_list):
    mse_list = []
    for data in data_list:
        deconvolved = data['deconvolver']
        mse = np.amin(deconvolved.deconvolved_mse_)
        norm = 1.1*np.amax(deconvolved.ground_truth_y)
        mse_list.append(mse/norm)
    ax.loglog(num_electrons, mse_list)


def _fwhm_plot(ax, num_electrons, data_list):
    fwhm_list = []
    for data in data_list:
        deconvolved = data['deconvolver']
        fwhm = _approximate_fwhm(deconvolved.deconvolved_x, deconvolved.deconvolved_y_)
        fwhm_list.append(fwhm)
    ax.semilogx(num_electrons, fwhm_list, color='r')
    ax.axhline(0.08325, linestyle='--', color='k')

def _format_figure(axs, data_list):
    axs[0].set_xlim((771, 779))
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_ylabel('Deconvolved\nMSE')
    axs[2].set_xscale('log')
    axs[2].set_xlabel('Detected Electrons')
    axs[2].set_ylabel('FWHM of First\nFeature')
    plt.tight_layout()

def _approximate_fwhm(deconvolved_x, deconvolved_y, center=0.0, width=1.0):
    loss = deconvolved_x-778
    loss_in_range = [(loss > (center-width/2)) & (loss < (center+width/2))]
    peak_location = loss[loss_in_range][np.argmax(deconvolved_y[loss_in_range])]
    peak_height = np.amax(deconvolved_y[loss_in_range])
    spec_below = deconvolved_y[loss < peak_location]
    loss_below = loss[loss < peak_location]
    above_peak = deconvolved_y[loss > peak_location]
    loss_above = loss[loss > peak_location]
    below_less_than_half = loss_below[spec_below < (peak_height/2)]
    below_hwhm = peak_location-below_less_than_half[-1]
    above_less_than_half = loss_above[above_peak < (peak_height/2)]
    above_hwhm = above_less_than_half[0]-peak_location
    fwhm = above_hwhm+below_hwhm
    return fwhm

def _label_spectra(ax, num_electrons):
    pass