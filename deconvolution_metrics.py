"""scikit-learn-style metrics for deconvolution results
"""

import numpy as np
from sklearn.metrics import mean_squared_error

def neg_deconvolved_mse(deconvolver, X, y=None):
    return -1*mean_squared_error(deconvolver.ground_truth_y, deconvolver.predict(X))

def neg_reconvolved_mse(deconvolver, X, y=None):
    return -1*mean_squared_error(np.mean(X, axis=0), deconvolver.reconvolved_y_)