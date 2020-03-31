"""Summary plot of deconvolution dependence on regularization parameter
"""

import matplotlib.pyplot as plt

def make_plot(deconvolved_gs):
    """Make summary plot of deconvolution dependence on regularization parameter
    """
    f = plt.figure()
    grid = plt.GridSpec(3, 2)
    ax_deconvolved_mse = f.add_subplot(grid[0, 0])
    ax_reconvolved_mse = f.add_subplot(grid[1, 0], sharex=ax_deconvolved_mse)
    ax_cv = f.add_subplot(grid[2, 0], sharex=ax_deconvolved_mse)
    ax_spectra = f.add_subplot(grid[:, 1])
    _make_deconvolved_mse_plot(ax_deconvolved_mse, deconvolved_gs)
    _make_reconvolved_mse_plot(ax_reconvolved_mse, deconvolved_gs)
    _make_cv_plot(ax_cv, deconvolved_gs)
    #_make_spectra_plot(ax_spectra, regulari)
    _format_plot(ax_deconvolved_mse, ax_reconvolved_mse, ax_cv, ax_spectra)

def _make_deconvolved_mse_plot(ax, deconvolved_gs):
    pass

def _make_cv_plot(ax, deconvolved_gs):
    ax.errorbar(deconvolved_gs.param_grid['regularizer_width'], -1*deconvolved_gs.cv_results_['mean_test_score'], deconvolved_gs.cv_results_['std_test_score'])

def _make_reconvolved_mse_plot(ax, deconvolved_gs):
    ax.errorbar(deconvolved_gs.param_grid['regularizer_width'], -1*deconvolved_gs.cv_results_['mean_train_score'], deconvolved_gs.cv_results_['std_train_score'])

def _format_plot(ax_deconvolved_mse, ax_reconvolved_mse, ax_cv, ax_spectra):
    ax_reconvolved_mse.set_xscale('log')
    ax_cv.set_xscale('log')