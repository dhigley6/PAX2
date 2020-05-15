#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley

Make plot of simulated deconvolution result
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_result(deconvolved):
    """Summarize deconvolution results
    deconvolved: fitted deconvolver
    """
    _, axs = plt.subplots(2, 1)
    _make_deconvolved_plot(axs[0], deconvolved)
    _make_reconvolved_plot(axs[1], deconvolved)
    _format_plot(axs)

def _make_deconvolved_plot(ax, deconvolved):
    ax.plot(deconvolved.deconvolved_x, deconvolved.ground_truth_y, label='Ground Truth')
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_, label='Deconvolved')

def _make_reconvolved_plot(ax, deconvolved):
    ax.plot(deconvolved.reconvolved_x, deconvolved.measured_y_, label='Measured')
    ax.plot(deconvolved.reconvolved_x, deconvolved.reconstruction_y_, label='Reconvolved')

def _format_plot(axs):
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Electron Energy (eV)')
    axs[1].set_ylabel('Intensity (a.u.)')
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.tight_layout()