"""Make plot showing simulated PAX performance on model Schlappa RIXS
with Ag 3d converter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

import pax_simulation_analysis
import visualize.set_plot_params
visualize.set_plot_params.init_paper_small()

FIGURES_DIR = 'figures'
LOG10_COUNTS_LIST = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]

SCHLAPPA_PARAMETERS = {
    'energy_loss': np.arange(-8, 10, 0.01),
    'iterations': int(1E5),
    'simulations': 1000,
    'cv_fold': 3,
    'regularizer_widths': np.logspace(-3, -1, 10)
}

def make_figure():
    log10_counts = LOG10_COUNTS_LIST
    data_list = []
    num_counts = []
    for i in log10_counts:
        data_list.append(pax_simulation_analysis.load(i, rixs='schlappa', photoemission='ag'))
        num_counts.append(10**i)
    spectra_log10_counts = [7.0, 5.0, 3.0]
    spectra_data_list = []
    spectra_num_counts = []
    for i in spectra_log10_counts:
        spectra_data_list.append(pax_simulation_analysis.load(i, rixs='schlappa', photoemission='ag'))
        spectra_num_counts.append(i)
    f = plt.figure(figsize=(3.37, 5.5))
    grid = plt.GridSpec(4, 2)
    ax_spectra = f.add_subplot(grid[:2, :])
    ax_deconvolved_mse = f.add_subplot(grid[2, :])
    ax_fwhm = f.add_subplot(grid[3, :], sharex=ax_deconvolved_mse)
    _spectra_plot(ax_spectra, spectra_data_list)
    _rmse_plot(ax_deconvolved_mse, num_counts, data_list)
    _fwhm_plot(ax_fwhm, num_counts, data_list)
    axs = [ax_spectra, ax_deconvolved_mse, ax_fwhm]
    _format_figure(axs, spectra_num_counts)
    file_name = f'{FIGURES_DIR}/pax_performance1.eps'
    plt.savefig(file_name, dpi=600)
    
def _spectra_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        deconvolved = data['deconvolver']
        energy_loss = -1*(deconvolved.deconvolved_x-778)
        offset = ind*1.0
        norm = 1.1*np.amax(deconvolved.ground_truth_y)
        ax.plot(energy_loss, offset+deconvolved.deconvolved_y_/norm, 'r', label='Deconvolved')
        ax.plot(energy_loss, offset+deconvolved.ground_truth_y/norm, 'k--', label='Ground Truth')
        
def _rmse_plot(ax, num_electrons, data_list):
    norm_rmse_list = []
    for data in data_list:
        data = data['deconvolver']
        deconvolved_mse = mean_squared_error(data.deconvolved_y_, data.ground_truth_y)
        rmse = np.sqrt(deconvolved_mse)
        norm_rmse = rmse/np.amax(data.ground_truth_y)
        norm_rmse_list.append(norm_rmse)
    ax.loglog(num_electrons, norm_rmse_list, color='r')
    
def _fwhm_plot(ax, num_electrons, data_list):
    fwhm_list = []
    for data in data_list:
        deconvolved = data['deconvolver']
        fwhm = _approximate_fwhm(deconvolved.deconvolved_x, deconvolved.deconvolved_y_)
        fwhm_list.append(fwhm)
    ax.semilogx(num_electrons, fwhm_list, color='r')
    ax.axhline(0.08325, linestyle='--', color='k')

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

def _format_figure(axs, spectra_counts):
    axs[0].set_xlim((-1, 7))
    axs[0].invert_xaxis()
    axs[0].set_ylim((-0.2, 3.5))
    axs[2].set_ylim((0, 0.4))
    axs[0].set_xlabel('Energy Loss (eV)')
    axs[0].set_ylabel('Intensity')
    axs[1].set_ylabel('Norm. RMS\nError')
    axs[2].set_ylabel('FWHM of\nFirst Peak')
    axs[2].set_xlabel('Detected Electrons')
    plt.setp(axs[1].get_xticklabels(), visible=False)
    legend_elements = [Line2D([0], [0], color='k', linestyle='--', label='Ground Truth'),
                       Line2D([0], [0], color='r', label='Deconvolved')]
    axs[0].legend(handles=legend_elements, loc='upper left', frameon=False)
    axs[0].text(-0.25, 2.3, '10$^'+str(int(spectra_counts[2]))+'$', ha='center', transform=axs[0].transData)
    axs[0].text(-0.25, 1.3, '10$^'+str(int(spectra_counts[1]))+'$', ha='center', transform=axs[0].transData)
    axs[0].text(-0.25, 0.3, '10$^'+str(int(spectra_counts[0]))+'$', ha='center', transform=axs[0].transData)
    plt.tight_layout()
    axs[0].text(0.9, 0.9, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.8, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    axs[2].text(0.9, 0.8, 'C', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[2].transAxes)