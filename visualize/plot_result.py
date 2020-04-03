#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley

Make plot of simulated deconvolution result
"""

import numpy as np
import matplotlib.pyplot as plt

def make_plot(deconvolved, pax_spectra, xray_xy):
    """Summarize deconvolution results
    deconvolved: fitted deconvolver
    sim: simulated PAX data set
    """
    _, axs = plt.subplots(2, 1)
    _make_deconvolved_plot(axs[0], deconvolved, xray_xy)
    _make_reconvolved_plot(axs[1], deconvolved, pax_spectra)
    _format_plot(axs)

def _make_deconvolved_plot(ax, deconvolved, xray_xy):
    ax.plot(xray_xy['x'], xray_xy['y'], label='Ground Truth')
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_, label='Deconvolved')

def _make_reconvolved_plot(ax, deconvolved, pax_spectra):
    ax.plot(pax_spectra['x'], np.mean(pax_spectra['y'], axis=0), label='PAX')
    ax.plot(deconvolved.reconvolved_x, deconvolved.reconvolved_y_, label='Reconvolved')

def _format_plot(axs):
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Electron Energy (eV)')
    axs[1].set_ylabel('Intensity (a.u.)')
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.tight_layout()