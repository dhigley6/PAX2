"""Plot 
"""

import matplotlib.pyplot as plt


def make_plot(deconvolved):
    plt.figure()
    plt.plot(deconvolved.impulse_response_x, deconvolved.impulse_response_y)