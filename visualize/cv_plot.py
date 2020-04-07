"""Summary plot of deconvolution dependence on regularization parameter
"""

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
    ax.errorbar(deconvolved_grid.regularizer_widths, deconvolved_grid.deconvolved_mse_, deconvolved_grid.deconvolved_mse_std_)

def _make_cv_plot(ax, deconvolved_grid):
    ax.errorbar(deconvolved_grid.regularizer_widths, deconvolved_grid.cv_, deconvolved_grid.cv_std_)

def _make_reconvolved_mse_plot(ax, deconvolved_grid):
    ax.errorbar(deconvolved_grid.regularizer_widths, deconvolved_grid.reconvolved_mse_, deconvolved_grid.reconvolved_mse_std_)

def _format_plot(ax_deconvolved_mse, ax_reconvolved_mse, ax_cv, ax_spectra):
    ax_reconvolved_mse.set_xscale('log')
    ax_cv.set_xscale('log')