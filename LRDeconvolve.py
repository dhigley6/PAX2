"""
Lucy-Richardson deconvolution and variants
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve

class LRDeconvolve(BaseEstimator):
    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, iterations=5):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.reconvolved_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.iterations = iterations

    def fit(self, X):
        """Perform PAX 
        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        measured_y = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR(measured_y)
        self.reconvolved_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
        return self

    def _LR(self, measured_y):
        """Perform Lucy-Richardson deconvolution of measured_y
        """
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        for _ in range(self.iterations):
            I = convolve(previous_O, self.impulse_response_y, mode='valid')
            relative_blur = measured_y/I
            correction_factor_estimate = convolve(relative_blur, impulse_response_y_reversed, mode='valid')
            current_O = previous_O*correction_factor_estimate
            previous_O = current_O
        return current_O

    def _deconvolution_guess(self, measured_y):
        """Return initial guess for deconvolved signal
        """
        sigma = 1
        x = self.impulse_response_x
        mu = np.mean(x)
        gauss = np.exp((-1/2)*((x-mu)/sigma)**2)
        gauss = gauss/np.sum(gauss)
        return convolve(measured_y, gauss, mode='valid')

    def predict(self, X):
        return self.deconvolved_y_
        
    def score(self, X_test):
        mean_X_test = np.mean(X_test, axis=0)
        mse = mean_squared_error(self.reconvolved_y_, mean_X_test)
        return -1*mse

    def _get_deconvolved_x(self):
        average_impulse_x = np.mean(self.impulse_response_x)
        deconvolved_x = self.convolved_x-average_impulse_x
        return deconvolved_x

class LRFisterDeconvolve(LRDeconvolve):
    """Lucy-Richardson deconvolution with Fister regularization
    """

    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, regularizer_width=0.05, iterations=5):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.reconvolved_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.regularizer_width = regularizer_width
        self.iterations = int(iterations)

    def _LR_fister(self, measured_y):
        """Perform Fister-regularized Lucy-Richardson deconvolution of measured_y
        """
        gauss = self._normalized_gaussian(
            self.impulse_response_x,
            np.mean(self.impulse_response_x),
            self.regularizer_width
        )
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        for _ in range(self.iterations):
            I = convolve(previous_O, self.impulse_response_y, mode='valid')
            relative_blur = measured_y/I
            correction_factor_estimate = convolve(relative_blur, impulse_response_y_reversed, mode='valid')
            current_O = previous_O*correction_factor_estimate
            current_O = convolve(current_O, gauss, mode='valid')
            previous_O = current_O
        return current_O

    def fit(self, X):
        """Perform PAX 
        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        measured_y = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR_fister(measured_y)
        self.reconvolved_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
        return self

    def _normalized_gaussian(self, x, mu, sigma):
        """Return a normalized gaussian function 
        center: mu
        standard deviation: sigma
        unit integrated amplitude
        """
        norm_gauss = np.exp((-1/2)*((x-mu)/sigma)**2)
        norm_gauss = norm_gauss/np.sum(norm_gauss)
        return norm_gauss

class LRL2Deconvolve(LRDeconvolve):
    """Lucy-Richardson deconvolution with L2 regularization
    """

    def __init__(self, impulse_response_x, impulse_response_y, alpha=0.05, iterations=5):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.alpha = alpha
        self.iterations = iterations

    def fit(self, X):
        """Perform PAX 
        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        measured_y = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LRL2(measured_y)
        self.reconvolved_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
        return self

    def _LRL2(self, measured_y):
        """Perform L2-regularized Lucy-Richardson deconvolution of measured_y
        """
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        for _ in range(self.iterations):
            I = convolve(previous_O, self.impulse_response_y, mode='valid')
            relative_blur = measured_y/I
            correction_factor_estimate = convolve(relative_blur, impulse_response_y_reversed, mode='valid')
            current_O = previous_O*correction_factor_estimate
            current_O = (-1+np.sqrt(1+2*self.alpha*current_O))/self.alpha
            previous_O = current_O
        return current_O