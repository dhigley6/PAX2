"""Plot photoemission spectrum used in PAX deconvolutions
"""

from typing import Union
import numpy as np
import matplotlib.pyplot as plt

from pax_deconvolve.deconvolution import deconvolvers


def plot_photoemission(
    deconvolved: Union[
        deconvolvers.LRFisterGrid,
        deconvolvers.LRFisterDeconvolve,
        deconvolvers.LRDeconvolve,
    ]
):
    """Plot photoemission spectrum used in deconvolution

    Parameters:
    -----------
    deconvolved: Fitted instance of deconvolver in
        pax_deconvolve.deconvolution.deconvolvers
    """
    binding_energy = -1 * np.flipud(deconvolved.impulse_response_x)
    photoemission = np.flipud(deconvolved.impulse_response_y)
    plt.figure()
    plt.plot(binding_energy, photoemission)
    _format_plot()


def _format_plot():
    """Format photoemission plot"""
    plt.xlabel("Binding Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
