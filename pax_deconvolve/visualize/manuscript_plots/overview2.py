"""
Make an overview plot of how PAX works for Ag 3d and Fermi edge cases
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects

from pax_deconvolve.pax_simulations import model_photoemission, model_rixs, simulate_pax
from pax_deconvolve.visualize import set_plot_params
set_plot_params.init_paper_small()

def make_plot():
    incident_photon_energy = 778
    energy_loss = np.arange(-20, 20, 0.005)
    schlappa_rixs = model_rixs.make_model_rixs('schlappa', energy_loss, incident_photon_energy)
    ag_photoemission = model_photoemission.make_model_photoemission('ag', schlappa_rixs, schlappa_rixs['x'][1]-schlappa_rixs['x'][0])
    fermi_photoemission = model_photoemission.make_model_photoemission('fermi', schlappa_rixs, schlappa_rixs['x'][1]-schlappa_rixs['x'][0])
    _, pax_spectra = simulate_pax.simulate(schlappa_rixs, ag_photoemission, 1E10, 1)
    pax_spectrum = {
        'x': pax_spectra['x'],
        'y': pax_spectra['y'][0]}
    _, fermi_pax_spectra = simulate_pax.simulate(schlappa_rixs, fermi_photoemission, 1E10, 1)
    fermi_pax_spectrum = {
        'x': fermi_pax_spectra['x'],
        'y': fermi_pax_spectra['y'][0]
    }
    f = plt.figure(figsize=(3.37, 4), constrained_layout=True)
    g = gridspec.GridSpec(ncols=2, nrows=3, figure=f)
    ax_rixs = f.add_subplot(g[0, :])
    ax_ag = f.add_subplot(g[1, 0])
    ax_fermi = f.add_subplot(g[1, 1])
    ax_ag_pax = f.add_subplot(g[2, 0])
    ax_fermi_pax = f.add_subplot(g[2, 1])
    ax_rixs.plot(schlappa_rixs['x'], schlappa_rixs['y'], color='k')
    ax_ag.plot(ag_photoemission['x'], ag_photoemission['y'], color='k')
    ax_fermi.plot(fermi_photoemission['x'], fermi_photoemission['y'], color='k')
    ax_ag_pax.plot(pax_spectrum['x'], pax_spectrum['y'], color='k')
    ax_fermi_pax.plot(fermi_pax_spectrum['x'], fermi_pax_spectrum['y'], color='k')
    ax_rixs.set_xlim((770, 780))
    ax_ag.set_xlim((380, 365))
    ax_ag_pax.set_xlim((395, 412))
    ax_fermi.set_xlim((5, -5))
    ax_fermi_pax.set_xlim((770, 780))

    for ax in [ax_rixs, ax_ag, ax_fermi, ax_ag_pax, ax_fermi_pax]:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    ax_rixs.set_xlabel('.', color=(0, 0, 0, 0))
    ax_ag_pax.set_xlabel('.', color=(0, 0, 0, 0))
    ax_fermi_pax.set_xlabel('.', color=(0, 0, 0, 0))
    ax_ag.set_xlabel('.', color=(0, 0, 0, 0))
    ax_fermi.set_xlabel('.', color=(0, 0, 0, 0))
    ax_rixs.set_ylabel('.', color=(0, 0, 0, 0))
    ax_ag.set_ylabel('Intensity')
    ax_ag_pax.set_ylabel('.', color=(0, 0, 0, 0))
    f.text(0.5, 0.02, 'Kinetic Energy (eV)', horizontalalignment='center')
    f.text(0.5, 0.36, 'Binding Energy (eV)', horizontalalignment='center')
    f.text(0.5, 0.69, 'Photon Energy (eV)', horizontalalignment='center')
    txt = f.text(0.3, 0.95, 'Desired RIXS\nr(E)', transform=ax_rixs.transAxes,
                horizontalalignment='center', verticalalignment='top', 
                fontsize=8, bbox=dict(facecolor='white', edgecolor='none', pad=0))
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    txt = f.text(0.525, 0.6, 'Photoemission\np(E)', transform=f.transFigure,
                horizontalalignment='center', verticalalignment='top', fontsize=8)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    txt = f.text(0.55, 0.25, 
       ''.join(['Measured PAX\n', r'$m(E) = r(E)\ast p(-E)$']), 
       transform=f.transFigure,
       horizontalalignment='center', verticalalignment='top', fontsize=8)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
    ax_rixs.text(0.9, 0.8, 'A', transform=ax_rixs.transAxes)
    ax_ag.text(0.1, 0.8, 'B', transform=ax_ag.transAxes)
    ax_fermi.text(0.9, 0.8, 'C', transform=ax_fermi.transAxes)
    ax_ag_pax.text(0.1, 0.8, 'D', transform=ax_ag_pax.transAxes)
    ax_fermi_pax.text(0.9, 0.8, 'E', transform=ax_fermi_pax.transAxes)
    plt.savefig('figures/2020_05_04_overview.eps', dpi=600)

def format_plot(f, axs):
    axs[0].yaxis.set_ticklabels([])
    axs[0].yaxis.set_ticks([])
    axs[1].yaxis.set_ticklabels([])
    axs[1].yaxis.set_ticks([])
    axs[2].yaxis.set_ticks([])
    axs[2].yaxis.set_ticklabels([])
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[1].set_xlabel('Binding Energy (eV)')
    axs[2].set_xlabel('Kinetic Energy (eV)')
    axs[1].set_ylabel('Intensity')
    #axs[0].set_xlim((776.5, 778.5))
    #axs[1].set_xlim((365, 380))
    #axs[2].set_xlim((400, 412))
    axs[0].text(0.9, 0.8, 'A', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.8, 'B', transform=axs[1].transAxes)
    axs[2].text(0.9, 0.8, 'C', transform=axs[2].transAxes)
    axs[0].text(0.3, 0.95, 'Desired RIXS\nr(E)', transform=axs[0].transAxes,
                horizontalalignment='center', verticalalignment='top', 
                fontsize=8)
    axs[1].text(0.55, 0.95, 'Photoemission\np(E)', transform=axs[1].transAxes,
                horizontalalignment='center', verticalalignment='top', fontsize=8)
    axs[2].text(0.35, 0.95, 
       ''.join(['Measured PAX\n', r'$m(E) = r(E)\ast p(-E)$']), 
       transform=axs[2].transAxes,
       horizontalalignment='center', verticalalignment='top', fontsize=8)
    #axs[1].text(0.9, 0.7, ')
    plt.tight_layout(w_pad=0, h_pad=0)