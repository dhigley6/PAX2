"""Make plot of statistics of deconvolved signal as a function of regularization hyperparameter
"""

import numpy as np 
import matplotlib.pyplot as plt
import datetime

from pax_deconvolve.pax_simulations import simulate_pax
from pax_deconvolve import pax_simulation_pipeline
from pax_deconvolve.visualize import set_plot_params
set_plot_params.init_paper_small()
from pax_deconvolve.visualize.manuscript_plots import schlappa_performance

FIGURES_DIR = 'figures'
LOG10_ELECTRONS_TO_PLOT = [3.0, 5.0, 7.0]

# Only show data with an index of the regularization strength at or above the below index
# This is so that Fister regularization strengths that correspond to convolving with Gaussians
# that are small with respect to a pixel (and therefore don't really make sense) are not
# included.
START_REG = 1

def make_figure():
    deconvolved_list, _, deconvolved_labels = _load_data()
    val_pax_spectrum = _make_example_pax_val_data()
    deconvolved_norm = np.amax(deconvolved_list[0].ground_truth_y)
    pax_norm = np.amax(deconvolved_list[0].measured_y_)
    _, axs = plt.subplots(3, 2, sharex='none', figsize=(6, 5))
    _single_deconvolved_plot(axs[0, 0], deconvolved_list[1], deconvolved_norm)
    _single_train_reconstruction_plot(axs[1, 0], deconvolved_list[1], pax_norm)
    _single_val_reconstruction_plot(axs[2, 0], deconvolved_list[2], val_pax_spectrum, pax_norm)
    _deconvolved_mse_plot(axs[0, 1], deconvolved_list, deconvolved_labels, deconvolved_norm)
    _reconvolved_mse_plot(axs[1, 1], deconvolved_list, deconvolved_labels, pax_norm)
    _cv_plot(axs[2, 1], deconvolved_list, deconvolved_labels, pax_norm)
    _format_figure(axs)
    file_name = f'{FIGURES_DIR}/effect_of_regularization_quant.eps'
    plt.savefig(file_name, dpi=600)

def _make_example_pax_val_data():
    _, example_spectra, _ = simulate_pax.simulate_from_presets(5.0, 'schlappa', 'ag', 1000, schlappa_performance.SCHLAPPA_PARAMETERS['energy_loss'])
    return np.mean(example_spectra['y'], axis=0)

def _load_data():
    deconvolved_list = []
    pax_spectra_list = []
    deconvolved_labels = []
    for i in LOG10_ELECTRONS_TO_PLOT:
        data = pax_simulation_pipeline.load(i, rixs='schlappa', photoemission='ag')
        deconvolved_list.append(data['deconvolver'])
        pax_spectra_list.append(data['pax_spectra'])
        deconvolved_labels.append('10$^'+str(int(i))+'$')
    return deconvolved_list, pax_spectra_list, deconvolved_labels

def _single_deconvolved_plot(ax, deconvolved, norm):
    ax.plot(deconvolved.deconvolved_x, deconvolved.ground_truth_y/norm, 'k--', label='Ground Truth')
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_/norm, 'r', label='Deconvolved')

def _single_train_reconstruction_plot(ax, deconvolved, norm):
    ax.plot(deconvolved.convolved_x, deconvolved.measured_y_/norm, 'k--', label='Train. PAX Data')
    ax.plot(deconvolved.convolved_x, deconvolved.reconvolved_y_/norm, 'r', label='Reconstruction')

def _single_val_reconstruction_plot(ax, deconvolved, val_pax_spectrum, norm):
    ax.plot(deconvolved.convolved_x, val_pax_spectrum/norm, 'k--', label='Val. PAX Data')
    ax.plot(deconvolved.convolved_x, deconvolved.reconvolved_y_/norm, 'r', label='Reconstruction')
    
def _deconvolved_mse_plot(ax, deconvolved_list, deconvolved_labels, norm):
    for deconvolved, deconvolved_label in zip(deconvolved_list, deconvolved_labels):
        deconvolved_rmse = np.sqrt(deconvolved.deconvolved_mse_)/norm
        line = ax.loglog(deconvolved.regularizer_widths[START_REG:], deconvolved_rmse[START_REG:], label=deconvolved_label)
        min_ind = np.argmin(deconvolved.deconvolved_mse_[START_REG:])
        ax.loglog(deconvolved.regularizer_widths[START_REG:][min_ind], deconvolved_rmse[START_REG:][min_ind], marker='x', color=line[0].get_color())

def _reconvolved_mse_plot(ax, deconvolved_list, deconvolved_labels, norm):
    for deconvolved, deconvolved_label in zip(deconvolved_list, deconvolved_labels):
        reconvolved_rmse = np.sqrt(deconvolved.reconvolved_mse_)/norm
        line = ax.loglog(deconvolved.regularizer_widths[START_REG:], reconvolved_rmse[START_REG:], label=deconvolved_label)
        min_ind = np.argmin(deconvolved.reconvolved_mse_[START_REG:])
        ax.loglog(deconvolved.regularizer_widths[START_REG:][min_ind], reconvolved_rmse[START_REG:][min_ind], marker='x', color=line[0].get_color())

def _cv_plot(ax, deconvolved_list, deconvolved_labels, norm):
    for deconvolved, deconvolved_label in zip(deconvolved_list, deconvolved_labels):
        cv_rmse = np.sqrt(deconvolved.cv_)/norm
        line = ax.loglog(deconvolved.regularizer_widths[START_REG:], cv_rmse[START_REG:]-np.amin(cv_rmse[START_REG:])+1E-7, label=deconvolved_label)
        min_ind = np.argmin(deconvolved.cv_[START_REG:])
        ax.loglog(deconvolved.regularizer_widths[START_REG:][min_ind], cv_rmse[START_REG:][min_ind]-np.amin(cv_rmse[START_REG:])+1E-7, marker='x', color=line[0].get_color())

def _format_figure(axs):
    axs[1, 0].set_xlim((396, 414))
    axs[2, 0].set_xlim((396, 414))
    axs[0, 1].yaxis.tick_right()
    axs[0, 1].yaxis.set_label_position('right')
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].yaxis.set_label_position('right')
    axs[2, 1].yaxis.tick_right()
    axs[2, 1].yaxis.set_label_position('right')
    axs[0, 0].set_xlim((771, 779))
    axs[0, 0].set_xlabel('Photon Energy (eV)')
    axs[0, 0].set_ylabel('Intensity (a.u.)')
    axs[1, 0].set_xlabel('Kinetic Energy (eV)')
    axs[1, 0].set_ylabel('Intensity (a.u.)')
    axs[2, 0].set_xlabel('Kinetic Energy (eV)')
    axs[2, 0].set_ylabel('Intensity (a.u.)')
    axs[0, 1].set_ylabel('Deconvolved\nRMSE')
    axs[1, 1].set_ylabel('Train.\nReconstruction RMSE')
    axs[2, 1].set_ylabel('Val. Reconstruction\nRMSE-minimum+10$^{-7}$')
    axs[2, 1].set_xlabel('Regularization Hyperparameter (eV)')
    axs[2, 1].set_xscale('log')
    axs[0, 0].text(0.9, 0.8, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.9, 0.2, 'B', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.9, 0.8, 'C', fontsize=10, weight='bold', horizontalalignment='center',
                  transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.9, 0.2, 'D', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[1, 1].transAxes)
    axs[2, 0].text(0.9, 0.8, 'E', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[2, 0].transAxes)
    axs[2, 1].text(0.9, 0.2, 'F', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[2, 1].transAxes)
    plt.tight_layout(h_pad=0)
    axs[0, 1].legend(loc='upper right', fontsize=9, frameon=False, handlelength=0.8, title='Detected Electrons', ncol=3, columnspacing=0.5)
    axs[0, 0].legend(loc='best')
    axs[1, 0].legend(loc='best')
    axs[2, 0].legend(loc='best')
