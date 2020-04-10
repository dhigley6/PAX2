"""Plot photoemission spectrum used in PAX deconvolutions
"""

import numpy as np
import matplotlib.pyplot as plt


def make_plot(deconvolved):
    binding_energy = -1*deconvolved.impulse_response_x
    photoemission = np.flipud(deconvolved.impulse_response_y)
    plt.figure()
    plt.plot(binding_energy, photoemission)
    _format_plot()

def _format_plot():
    plt.xlabel('Binding Energy (eV)')
    plt.ylabel('Intensity (a.u.)')