"""Summary plot of deconvolution dependence on regularization parameter
"""

import numpy as np
import matplotlib.pyplot as plt

def make_plot(deconvolved_grid):
    """Make summary plot of deconvolution dependence on regularization parameter
    """
    f = plt.figure()
    grid = plt.GridSpec(3, 2)
    ax_deconvolved_mse = f.add_subplot(grid[0, 0])
    ax_reconvolved_mse = f.add_subplot(grid[1, 0], sharex=ax_deconvolved_mse)
    ax_cv = f.add_subplot(grid[2, 0], sharex=ax_deconvolved_mse)
    ax_spectra = f.add_subplot(grid[:, 1])
    _make_deconvolved_mse_plot(ax_deconvolved_mse, deconvolved_grid)
    _make_reconvolved_mse_plot(ax_reconvolved_mse, deconvolved_grid)
    _make_cv_plot(ax_cv, deconvolved_grid)
    #_make_spectra_plot(ax_spectra, regulari)
    _format_plot(ax_deconvolved_mse, ax_reconvolved_mse, ax_cv, ax_spectra)

def _make_deconvolved_mse_plot(ax, deconvolved_grid):
    line = ax.errorbar(deconvolved_grid.regularizer_widths, deconvolved_grid.deconvolved_mse_, deconvolved_grid.deconvolved_mse_std_)
    min_ind = np.argmin(deconvolved_grid.deconvolved_mse_)
    ax.plot(deconvolved_grid.regularizer_widths[min_ind], deconvolved_grid.deconvolved_mse_[min_ind], marker='x', color=line[0].get_color())

def _make_cv_plot(ax, deconvolved_grid):
    line = ax.errorbar(deconvolved_grid.regularizer_widths, deconvolved_grid.cv_, deconvolved_grid.cv_std_)
    min_ind = np.argmin(deconvolved_grid.cv_)
    ax.plot(deconvolved_grid.regularizer_widths[min_ind], deconvolved_grid.cv_[min_ind], marker='x', color=line[0].get_color())

def _make_reconvolved_mse_plot(ax, deconvolved_grid):
    line = ax.errorbar(deconvolved_grid.regularizer_widths, deconvolved_grid.reconvolved_mse_, deconvolved_grid.reconvolved_mse_std_)
    min_ind = np.argmin(deconvolved_grid.reconvolved_mse_)
    ax.plot(deconvolved_grid.regularizer_widths[min_ind], deconvolved_grid.reconvolved_mse_[min_ind], marker='x', color=line[0].get_color())

def _format_plot(ax_deconvolved_mse, ax_reconvolved_mse, ax_cv, ax_spectra):
    ax_deconvolved_mse.set_ylabel('Deconvolved MSE')
    ax_reconvolved_mse.set_ylabel('Reconvolved MSE')
    ax_cv.set_ylabel('CV')
    ax_cv.set_xlabel('Regularizer Width (eV)')
    ax_reconvolved_mse.set_xscale('log')
    ax_cv.set_xscale('log')