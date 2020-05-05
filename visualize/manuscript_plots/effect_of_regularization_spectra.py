#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dhigley
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

import visualize.set_plot_params
visualize.set_plot_params.init_paper_small()
import pax_simulation_analysis
from pax_simulations import simulate_pax
import LRDeconvolve

START_REGULARIZER = 0

def run_sim():
    parameters = pax_simulation_analysis.DEFAULT_PARAMETERS
    impulse_response, pax_spectra, xray_xy = simulate_pax.simulate_from_presets(
        4,
        'schlappa',
        'ag',
        parameters['simulations'],
        parameters['energy_loss']
    )
    regularizer_widths = parameters['regularizer_widths']
    regularizer_widths = np.append([0], regularizer_widths)
    iterations = 1E2
    results = Parallel(n_jobs=-1)(delayed(run_single_deconvolution)(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations) for regularizer_width in regularizer_widths)
    file_name = 'simulated_results/test.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump(results, f)

def load_sim():
    file_name = 'simulated_results/test.pickle'
    with open(file_name, 'rb') as f:
        results = pickle.load(f)
    return results
    
def run_single_deconvolution(impulse_response, pax_spectra, xray_xy, regularizer_width, iterations):
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
    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    log10_electrons_to_plot = [3.0, 5.0, 7.0]
    data_sets = []
    data_labels = []
    for i in log10_electrons_to_plot:
        data = pax_simulation_analysis.load(i)
        data_sets.append(data)
        data_labels.append('10$^'+str(int(i))+'$')
    _, axs = plt.subplots(3, 1, sharex=True, figsize=(3.37, 4))
    _deconvolved_mse_plot(axs[0], data_sets, data_labels)
    _reconvolved_mse_plot(axs[1], data_sets, data_labels)
    _cv_plot(axs[2], data_sets, data_labels)
    _format_figure(axs)
    file_name = f'{FIGURES_DIR}/effect_of_regularization_quant.eps'
    plt.savefig(file_name, dpi=600)
    
def _deconvolved_mse_plot(ax, data_sets, data_labels):
    for data, data_label in zip(data_sets, data_labels):
        deconvolved_mse = postprocess.deconvolved_mse_data_sets(data)
        line = ax.loglog(data[0]['regularizer_widths'][START_REGULARIZER:], deconvolved_mse,
                  label=data_label)
        min_ind = np.argmin(deconvolved_mse)
        ax.loglog(data[0]['regularizer_widths'][START_REGULARIZER:][min_ind],
                  deconvolved_mse[min_ind], marker='x',
                  color=line[0].get_color())
        
def _reconvolved_mse_plot(ax, data_sets, data_labels):
    for data, data_label in zip(data_sets, data_labels):
        reconvolved_mse = postprocess.reconvolved_mse_data_sets(data)
        line = ax.loglog(data[0]['regularizer_widths'][START_REGULARIZER:], reconvolved_mse,
                  label=data_label)
        min_ind = np.argmin(reconvolved_mse)
        ax.loglog(data[0]['regularizer_widths'][START_REGULARIZER:][min_ind],
                  reconvolved_mse[START_REGULARIZER:][min_ind], marker='x',
                  color=line[0].get_color())

def _cv_plot(ax, data_sets, data_labels):
    for data, data_label in zip(data_sets, data_labels):
        cv_set = postprocess.cv(data)
        line = ax.loglog(data[0]['regularizer_widths'][START_REGULARIZER:], cv_set,
           label=data_label)
        min_ind = np.argmin(cv_set)
        ax.loglog(data[0]['regularizer_widths'][START_REGULARIZER:][min_ind],
            cv_set[min_ind], marker='x',
            color=line[0].get_color())
    
def _format_figure(axs):
    axs[0].set_ylabel('MSE')
    axs[1].set_ylabel('Reconvolved\nMSE')
    axs[2].set_ylabel('CV')
    axs[2].set_xlabel('Regularization Parameter Width (eV)')
    axs[0].text(0.9, 0.2, 'A', fontsize=10, weight='bold', horizontalalignment='center',
                   transform=axs[0].transAxes)
    axs[1].text(0.9, 0.2, 'B', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[1].transAxes)
    axs[2].text(0.9, 0.2, 'C', fontsize=10, weight='bold', horizontalalignment='center',
       transform=axs[2].transAxes)
    plt.tight_layout(h_pad=0)
    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
    axs[1].legend(loc='upper left', fontsize=9, frameon=True,
       handlelength=0.8, title='Detected\nElectrons',
       bbox_to_anchor=(1, 0.5))