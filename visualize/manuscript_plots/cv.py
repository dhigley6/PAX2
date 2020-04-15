"""Make plot of statistics of deconvolved signal as a function of regularization hyperparameter
"""

import numpy as np 
import matplotlib.pyplot as plt 

from pax_simulations import run_analyze_save_load
import visualize.set_plot_params
visualize.set_plot_params.init_paper_small()

FIGURES_DIR = 'figures'
LOG10_ELECTRONS_TO_PLOT = [3.0, 5.0, 7.0]

def make_figure():
    deconvolved_list = []
    deconvolved_labels = []
    for i in LOG10_ELECTRONS_TO_PLOT:
        deconvolved = run_analyze_save_load.load(i, rixs='schlappa', photoemission='ag')['deconvolver']
        deconvolved_list.append(deconvolved)
        deconvolved_labels.append('10$^'+str(int(i))+'$')
    _, axs = plt.subplots(3, 2, sharex='none', figsize=(6, 5))
    _single_deconvolved_plot(axs[0, 0], deconvolved_list[0])
    _single_train_reconstruction_plot(axs[1, 0], deconvolved_list[0])
    _single_val_reconstruction_plot(axs[2, 0], deconvolved_list[0])
    _deconvolved_mse_plot(axs[0, 1], deconvolved_list, deconvolved_labels)
    _reconvolved_mse_plot(axs[1, 1], deconvolved_list, deconvolved_labels)
    _cv_plot(axs[2, 1], deconvolved_list, deconvolved_labels)
    _format_figure(axs)
    file_name = f'{FIGURES_DIR}/effect_of_regularization_quant.eps'
    plt.savefig(file_name, dpi=600)

def _single_deconvolved_plot(ax, deconvolved):
    ax.plot(deconvolved.deconvolved_x, deconvolved.ground_truth_y, 'k--', label='Ground Truth')
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_, 'r', label='Deconvolved')

def _single_train_reconstruction_plot(ax, deconvolved):
    ax.plot(deconvolved.convolved_x, deconvolved.measured_y_, 'k--', label='Train. PAX Data')
    ax.plot(deconvolved.convolved_x, deconvolved.reconvolved_y_, 'r', label='Reconstruction')

def _single_val_reconstruction_plot(ax, deconvolved):
    ax.plot(deconvolved.convolved_x, deconvolved.measured_y_, 'k--', label='Val. PAX Data')
    ax.plot(deconvolved.convolved_x, deconvolved.reconvolved_y_, 'r', label='Reconstruction')
    
def _deconvolved_mse_plot(ax, deconvolved_list, deconvolved_labels):
    for deconvolved, deconvolved_label in zip(deconvolved_list, deconvolved_labels):
        line = ax.loglog(deconvolved.regularizer_widths, deconvolved.deconvolved_mse_, label=deconvolved_label)
        min_ind = np.argmin(deconvolved.deconvolved_mse_)
        ax.loglog(deconvolved.regularizer_widths[min_ind], deconvolved.deconvolved_mse_[min_ind], marker='x', color=line[0].get_color())

def _reconvolved_mse_plot(ax, deconvolved_list, deconvolved_labels):
    for deconvolved, deconvolved_label in zip(deconvolved_list, deconvolved_labels):
        line = ax.loglog(deconvolved.regularizer_widths, deconvolved.reconvolved_mse_, label=deconvolved_label)
        min_ind = np.argmin(deconvolved.reconvolved_mse_)
        ax.loglog(deconvolved.regularizer_widths[min_ind], deconvolved.reconvolved_mse_[min_ind], marker='x', color=line[0].get_color())

def _cv_plot(ax, deconvolved_list, deconvolved_labels):
    for deconvolved, deconvolved_label in zip(deconvolved_list, deconvolved_labels):
        line = ax.loglog(deconvolved.regularizer_widths, deconvolved.cv_-np.amin(deconvolved.cv_)+1E-7, label=deconvolved_label)
        min_ind = np.argmin(deconvolved.cv_)
        ax.loglog(deconvolved.regularizer_widths[min_ind], deconvolved.cv_[min_ind]-np.amin(deconvolved.cv_)+1E-7, marker='x', color=line[0].get_color())

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
    axs[0, 1].set_ylabel('Deconvolved\nMSE')
    axs[1, 1].set_ylabel('Reconvolved\nMSE')
    axs[2, 1].set_ylabel('CV-min(CV)+10$^{-7}$')
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
