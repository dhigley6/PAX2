"""
Lucy-Richardson deconvolution and variants
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve

class LRDeconvolve(BaseEstimator):
    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, iterations=5, ground_truth_y=None):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.reconvolved_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.iterations = iterations
        self.writer = tf.summary.create_file_writer(f'logdir_test')
        self.ground_truth_y = ground_truth_y

    def fit(self, X):
        """Perform PAX 
        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        self.measured_y_ = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR(self.measured_y_)
        self.reconvolved_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
        return self

    def _LR(self, measured_y):
        """Perform Lucy-Richardson deconvolution of measured_y
        """
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        writer = tf.summary.create_file_writer('logdir_test')
        with writer.as_default():
            for iteration in range(self.iterations):
                I = convolve(previous_O, self.impulse_response_y, mode='valid')
                relative_blur = measured_y/I
                correction_factor_estimate = convolve(relative_blur, impulse_response_y_reversed, mode='valid')
                current_O = previous_O*correction_factor_estimate
                previous_O = current_O
                self._save_iteration_stats(current_O, iteration)
                writer.flush()
        return current_O

    def _save_iteration_stats(self, current_deconvolved, iteration):
        current_reconstruction = convolve(current_deconvolved, self.impulse_response_y, mode='valid')
        reconstruction_mse = mean_squared_error(current_reconstruction, self.measured_y_)
        tf.summary.scalar('reconstruction_mse', reconstruction_mse, step=iteration)
        if self.ground_truth_y is not None:
            # We have access to the ground truth, so we can calculate a few more metrics
            deconvolved_mse = mean_squared_error(current_deconvolved, self.ground_truth_y)
            tf.summary.scalar('deconvolved_mse', deconvolved_mse, step=iteration)
            ground_truth_reconvolved = convolve(self.ground_truth_y, self.impulse_response_y, mode='valid')
            reconvolved_mse = mean_squared_error(current_reconstruction, ground_truth_reconvolved)
            tf.summary.scalar('reconvolved_mse', reconvolved_mse, step=iteration)


    def _deconvolution_guess(self, measured_y):
        """Return initial guess for deconvolved signal
        """
        sigma = 1
        x = self.impulse_response_x
        mu = np.mean(x)
        gauss = np.exp((-1/2)*((x-mu)/sigma)**2)
        gauss = gauss/np.sum(gauss)
        #return convolve(measured_y, gauss, mode='valid')
        return measured_y

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

    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, regularizer_width=0.05, iterations=5, ground_truth_y=None):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.reconvolved_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.regularizer_width = regularizer_width
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y

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
        writer = tf.summary.create_file_writer(f'logdir_test/{self.regularizer_width}')
        with writer.as_default():
            for iteration in range(self.iterations):
                I = convolve(previous_O, self.impulse_response_y, mode='valid')
                relative_blur = measured_y/I
                correction_factor_estimate = convolve(relative_blur, impulse_response_y_reversed, mode='valid')
                current_O = previous_O*correction_factor_estimate
                current_O = convolve(current_O, gauss, mode='valid')
                previous_O = current_O
                self._save_iteration_stats(current_O, iteration)
                writer.flush()
        return current_O

    def fit(self, X):
        """Perform PAX 
        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        self.measured_y_ = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR_fister(self.measured_y_)
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