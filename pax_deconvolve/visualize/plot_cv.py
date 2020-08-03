"""Summary plot of deconvolution dependence on regularization parameter for
deconvolved data with known ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pax_deconvolve.deconvolution import deconvolvers


def plot_cv(deconvolved_grid: deconvolvers.LRFisterGrid):
    """Make summary plot of deconvolution dependence on regularization parameter

    Parameters:
    -----------
    deconvolved_grid: deconvolvers.LRFisterGrid
        Fitted instance of LRFisterGrid
    """
    _, axs = plt.subplots(3, 1, sharex=True)
    _make_deconvolved_mse_plot(axs[0], deconvolved_grid)
    _make_reconvolved_train_mse_plot(axs[1], deconvolved_grid)
    _make_reconvolved_test_mse_plot(axs[2], deconvolved_grid)
    _format_plot(axs)


def _make_deconvolved_mse_plot(ax: Axes, deconvolved_grid: deconvolvers.LRFisterGrid):
    """Plot deconvolved MSE vs regularization strength

    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        Axis to plot on
    deconvolved_grid: deconvolvers.LRFisterGrid
        Fitted instance of LRFisterGrid
    """
    line = ax.errorbar(
        deconvolved_grid.regularization_strengths,
        deconvolved_grid.deconvolved_mse_,
        deconvolved_grid.deconvolved_mse_std_,
    )
    min_ind = np.argmin(deconvolved_grid.deconvolved_mse_)
    ax.plot(
        deconvolved_grid.regularization_strengths[min_ind],
        deconvolved_grid.deconvolved_mse_[min_ind],
        marker="x",
        color=line[0].get_color(),
    )


def _make_reconvolved_test_mse_plot(
    ax: Axes, deconvolved_grid: deconvolvers.LRFisterGrid
):
    """Plot test reconvolved MSE vs regularization strength

    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        Axis to plot on
    deconvolved_grid: deconvolvers.LRFisterGrid
        Fitted instance of LRFisterGrid
    """
    line = ax.errorbar(
        deconvolved_grid.regularization_strengths,
        deconvolved_grid.cv_,
        deconvolved_grid.cv_std_,
    )
    min_ind = np.argmin(deconvolved_grid.cv_)
    ax.plot(
        deconvolved_grid.regularization_strengths[min_ind],
        deconvolved_grid.cv_[min_ind],
        marker="x",
        color=line[0].get_color(),
    )


def _make_reconvolved_train_mse_plot(
    ax: Axes, deconvolved_grid: deconvolvers.LRFisterGrid
):
    """Plot training reconvolved MSE vs regularization strength

    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        Axis to plot on
    deconvolved_grid: deconvolvers.LRFisterGrid
        Fitted instance of LRFisterGrid
    """
    line = ax.errorbar(
        deconvolved_grid.regularization_strengths,
        deconvolved_grid.reconstruction_train_mse_,
        deconvolved_grid.reconstruction_train_mse_std_,
    )
    min_ind = np.argmin(deconvolved_grid.reconstruction_train_mse_)
    ax.plot(
        deconvolved_grid.regularization_strengths[min_ind],
        deconvolved_grid.reconstruction_train_mse_[min_ind],
        marker="x",
        color=line[0].get_color(),
    )


def _format_plot(axs: np.ndarray):
    """Format figure

    Parameters:
    -----------
        axs: np.ndarray
            Array of axes of figure
    """
    axs[0].set_ylabel("Deconvolved MSE")
    axs[1].set_ylabel("Reconstruction\nTraining MSE")
    axs[2].set_ylabel("Reconstruction\nValidation MSE")
    axs[2].set_xlabel("Regularizer Width (eV)")
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
