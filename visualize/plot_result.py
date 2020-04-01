#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley

Make plot of simulated deconvolution result
"""

import matplotlib.pyplot as plt

def make_plot(deconvolved, sim):
    """Summarize deconvolution results
    deconvolved: fitted deconvolver
    sim: simulated PAX data set
    """
    _, axs = plt.subplots(2, 1)
    _make_deconvolved_plot(axs[0], deconvolved, sim)
    _make_reconvolved_plot(axs[1], deconvolved, sim)
    _format_plot(axs)

def _make_deconvolved_plot(ax, deconvolved, sim):
    ax.plot(sim['xray_xy']['x'], sim['xray_xy']['y'], label='Ground Truth')
    ax.plot(deconvolved.deconvolved_x, deconvolved.deconvolved_y_, label='Deconvolved')

def _make_reconvolved_plot(ax, deconvolved, sim):
    ax.plot(sim['mean_pax_xy']['x'], sim['mean_pax_xy']['y'], label='PAX')
    ax.plot(deconvolved.reconvolved_x, deconvolved.reconvolved_y_, label='Reconvolved')

def _format_plot(axs):
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Electron Energy (eV)')
    axs[1].set_ylabel('Intensity (a.u.)')
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.tight_layout()