#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

from pax_deconvolve.visualize import set_plot_params
set_plot_params.init_paper_small()
from pax_deconvolve import pax_simulation_pipeline
from pax_deconvolve.pax_simulations import simulate_pax
from pax_deconvolve import LRDeconvolve

START_REGULARIZER = 0

def run_sim():
    results_4 = _run_deconvolution_set(4)
    results_7 = _run_deconvolution_set(7)
    results = {
        '4': results_4,
        '7': results_7}
    file_name = 'pax_deconvolve/simulated_results/test.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)

def _run_deconvolution_set(log10_num_electrons):
    parameters = pax_simulation_pipeline.DEFAULT_PARAMETERS
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        log10_num_electrons,
        'schlappa',
        'ag',
        parameters['simulations'],
        parameters['energy_loss']
    )
    regularizer_widths = parameters['regularizer_widths']
    iterations = 1E5
    results = Parallel(n_jobs=-1)(delayed(_run_single_deconvolution)(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations) for regularizer_width in regularizer_widths)
    return results


def load_sim():
    file_name = 'pax_deconvolve/simulated_results/test.pickle'
    with open(file_name, 'rb') as f:
        results = pickle.load(f)
    return results
    
def _run_single_deconvolution(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations):
    deconvolver = LRDeconvolve.LRFisterDeconvolve(
        impulse_response['x'],
        impulse_response['y'],
        pax_spectra['x'],
        regularizer_width=regularizer_width,
        iterations=iterations,
        ground_truth_y=xray_xy['y']
    )
    deconvolver.fit(np.array(pax_spectra['y']))
    return deconvolver

def make_figure():
    results = load_sim()
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(3.37, 3.75))
    to_plot_4 = list(results['4'][i] for i in [2, 5, 9])
    to_plot_7 = list(results['7'][i] for i in [2, 5, 9])
    _regularization_offset_plot(axs[0], to_plot_4)
    _regularization_offset_plot(axs[1], to_plot_7)
    _format_figure(f, axs)
    plt.savefig('figures/effect_of_regularization_spectra.eps', dpi=600)

def _format_figure(f, axs):
    axs[0].set_xlim((771, 779))
    axs[0].set_ylim((-0.2, 4.5))
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[0].set_xlabel('.', color=(0, 0, 0, 0))
    f.text(0.57, 0.02, 'Photon Energy (eV)', horizontalalignment='center')
    axs[0].set_title('10$^4$\nElectrons')
    axs[1].set_title('10$^7$\nElectrons')
    axs[0].text(0.1, 0.9, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.9, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    plt.tight_layout()
    regularizer_widths = pax_simulation_pipeline.DEFAULT_PARAMETERS['regularizer_widths']
    regularizers_to_plot = list(regularizer_widths[i] for i in [2, 5, 9])
    regularizer_labels = [
        r'$\sigma = '+str(round(regularizers_to_plot[0]*1E3, 1))+'$ meV',
        r'$\sigma = '+str(round(regularizers_to_plot[1]*1E3, 1))+'$ meV',
        r'$\sigma = '+str(round(regularizers_to_plot[2]*1E3, 1))+'$ meV'
    ]
    regularizer_label_heights = [0.15, 0.43, 0.73]
    for regularizer_label, height in zip(regularizer_labels, regularizer_label_heights):
        axs[1].text(1.1, height, regularizer_label, ha='center', clip_on=False,
            bbox=dict(facecolor='white', edgecolor='none', pad=0),
            transform=axs[0].transAxes)
    big_ax = _make_big_dummy_ax(plt.gcf())
    big_ax.plot([], [], 'k--', label='Ground Truth')
    big_ax.plot([], [], 'r', label='Deconvolved')
    big_ax.legend(loc='upper center', frameon=True, borderpad=0.3,
                  framealpha=1, borderaxespad=0.3)
    big_ax.set_xticks([])
    big_ax.set_yticks([])

def _regularization_offset_plot(ax, data_list):
    for ind, data in enumerate(data_list):
        offset = 1.4*ind
        norm = 1.1*np.amax(data.ground_truth_y)
        ax.plot(data.deconvolved_x, offset+data.deconvolved_y_/norm, 'r', label='Deconvolved')
        ax.plot(data.deconvolved_x, offset+data.ground_truth_y/norm, 'k--', label='Ground Truth')

def _make_big_dummy_ax(fig):
    """Make a big dummy axis over the entire figure to use for global labels
    """
    dummyax = fig.add_subplot(111, frameon=False)
    dummyax.spines['top'].set_color('none')
    dummyax.spines['bottom'].set_color('none')
    dummyax.spines['left'].set_color('none')
    dummyax.spines['right'].set_color('none')
    dummyax.tick_params(labelcolor='none', top='off', left='off', right='off', bottom='off')
    plt.setp(dummyax.get_yticklabels(), alpha=0)
    return dummyax