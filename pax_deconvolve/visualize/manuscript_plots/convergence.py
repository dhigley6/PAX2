"""Make plot showing how we approximate that the desired level of convergence has been reached
"""

import numpy as np 
import matplotlib.pyplot as plt 

from pax_deconvolve.visualize import set_plot_params
set_plot_params.init_paper_small()

FIGURES_DIR = 'figures'
REGULARIZATION_PARAMETERS = [0.0028, 0.0046, 0.0077, 0.013]

def make_figure():
    _, axs = plt.subplots(2, 1, sharex=True, figsize=(3.37, 3.5))
    deconvolved_list = _load_deconvolved_mse()
    _make_deconvolved_mse_plot(axs[0], deconvolved_list)
    val_list = _load_val_mse()
    _make_val_mse_plot(axs[1], val_list)
    _format_figure(axs)
    file_name = f'{FIGURES_DIR}/convergence.eps'
    plt.savefig(file_name, dpi=600)

def _load_deconvolved_mse():
    data_list = []
    for par in REGULARIZATION_PARAMETERS:
        file_name = f'data/convergence_deconvolved/run-{str(par)}.csv'
        data = np.genfromtxt(file_name, skip_header=1, delimiter=',')
        data_list.append(data)
    return data_list

def _load_val_mse():
    data_list = []
    for par in REGULARIZATION_PARAMETERS:
        file_name = f'data/convergence_validation_reconstruction/run-{str(par)}.csv'
        data = np.genfromtxt(file_name, skip_header=1, delimiter=',')
        data_list.append(data)
    return data_list

def _make_deconvolved_mse_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        ax.plot(data[:, 1], data[:, 2], label=str(REGULARIZATION_PARAMETERS[ind]))

def _make_val_mse_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        ax.plot(data[:, 1], data[:, 2], label=str(REGULARIZATION_PARAMETERS[ind]))
    ax.axvline(77*4, color='k', linestyle='--')

def _format_figure(axs):
    axs[1].annotate('stopping\ncriterion met', (308, 0.106), xytext=(508, 0.106), arrowprops=dict(arrowstyle="->"), verticalalignment='center')
    axs[0].set_ylim((0.05, 0.35))
    axs[1].set_ylim((0.101, 0.108))
    axs[0].legend(loc='upper right', ncol=2, title='Regularization\nStrength (eV)')
    axs[0].set_ylabel('Deconvolved\nMSE (a.u.)')
    axs[1].set_ylabel('Val. Reconstruction\nMSE (a.u.)')
    axs[1].set_xlabel('Iterations')
    axs[0].text(0.9, 0.05, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.85, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    plt.tight_layout()