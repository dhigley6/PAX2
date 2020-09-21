#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley

Make plot of simulated deconvolution result (ground truth must be accessible)
"""

from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pax_deconvolve.deconvolution import deconvolvers


def plot_result(
    deconvolved: Union[
        deconvolvers.LRFisterGrid,
        deconvolvers.LRFisterDeconvolve,
        deconvolvers.LRDeconvolve,
    ]
):
    """Summarize deconvolution results

    Parameters:
    -----------
    deconvolved: Fitted instance of deconvolver in
        pax_deconvolve.deconvolution.deconvolvers
    """
    _, axs = plt.subplots(2, 1)
    _make_deconvolved_plot(axs[0], deconvolved)
    _make_reconvolved_plot(axs[1], deconvolved)
    _format_plot(axs)


def _make_deconvolved_plot(
    ax: Axes,
    deconvolved: Union[
        deconvolvers.LRFisterGrid,
        deconvolvers.LRFisterDeconvolve,
        deconvolvers.LRDeconvolve,
    ],
):
    """Plot deconvolved and ground truth spectrum

    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        Axis to plot on
    deconvolved: Fitted instance of deconvolver in
        pax_deconvolve.deconvolution.deconvolvers
    """
    if deconvolved.ground_truth_y is not None:
        ax.plot(deconvolved.deconvolved_x, deconvolved.ground_truth_y, label="Ground Truth")
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_, label="Deconvolved")


def _make_reconvolved_plot(
    ax: Axes,
    deconvolved: Union[
        deconvolvers.LRFisterGrid,
        deconvolvers.LRFisterDeconvolve,
        deconvolvers.LRDeconvolve,
    ],
):
    """Plot reconvolved spectrum and measured spectrum

    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        Axis to plot on
    deconvolved: Fitted instance of deconvolver in
        pax_deconvolve.deconvolution.deconvolvers
    """
    ax.plot(deconvolved.convolved_x, deconvolved.measured_y_, label="Measured")
    ax.plot(deconvolved.convolved_x, deconvolved.reconstruction_y_, label="Reconvolved")


def _format_plot(axs: np.ndarray):
    """Format plot

    parameters:
    -----------
    axs: np.ndarray
        Array of matplotlib.axes.Axes
    """
    axs[0].set_xlabel("Photon Energy (eV)")
    axs[0].set_ylabel("Intensity (a.u.)")
    axs[1].set_xlabel("Electron Energy (eV)")
    axs[1].set_ylabel("Intensity (a.u.)")
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    plt.tight_layout()
