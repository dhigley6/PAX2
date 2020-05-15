"""
Classes for regularized scaled gradient deconvolution
These are equivalent to regularized Lucy-Richardson deconvolution in the case
that the impulse response function is negligible at the boundaries.

The regularization we employed here was originally suggested by Fister et al.
for regularizing Lucy-Richardson deconvolution. This consists of convolving
the estimate of the deconvolved signal with a Gaussian after each iteration
of deconvolution. The width of the Gaussian sets the strength of the regularizaion.
See the below reference for more details:
T. T. Fister et al. Phys. Rev. B 75, 174106 (2007)
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve
from sklearn.model_selection import GridSearchCV

from pax_deconvolve.deconvolution import deconvolution_metrics

LOGDIR = 'logdir/'

class LRFisterGrid(BaseEstimator):
    """Fister-regularized deconvolution with regularization chosen by cross validation.

    Attributes:
        impulse_response_x {array-like of shape (n_i,)}: x-values (locations) 
            of impulse response
        impulse_response_y {array-like of shape (n_i,)}: y-values (intensities) 
            of impulse response
        convolved_x {array-like of shape (n_c,)}: x-values (locations) 
            of convolved data
        regularization_strengths {array-like of shape (n_regularizers,)}: 
            Regularization strengths to try
        iterations (int): Number of iterations to use in deconvolution
        ground_truth_y {array-like of shape (n_c,)}: y-values (intensities) 
            of ground truth for deconvolution (None if not available)
        cv_folds (int): Number of folds of cross validation to do
        best_regularization_strength_ {float}: Best regularization strength 
            found with cross validation
        measured_y_ {array-like of shape (n_c,)}: Average measurement
        deconvolved_y_ {array-like of shape (n_c,)}: Deconvolved intensities
        reconstruction_y_ {array-like of shape (n_c,)}: Reconstruction of
            input data from deconvolved result
        deconvolved_mse_ {array-like of shape (n_regularizers,)}: mean squared 
            error of deconvolved data as a function of regularization strength
        deconvolved_mse_std_ {array-like of shape (n_regularizers,)}: standard
            deviation of deconvolved_mse_
        reconstruction_train_mse_ {array-like of shape (n_regularizers,)}: mean
            squared deviation of reconstruction from average training data
        reconstruction_train_mse_std_ {array-like of shape (n_regularizers,)}:
            standard deviation of reconstruction_mse_
        cv_ {array-like of shape (n_regularizers,)}: mean squared deviation
            of reconstruction from average validation data
        cv_std_ {array-like of shape (n_regularizers,)}: standard deviation
            of cv_
    """

    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, 
                 regularization_strengths=[0.01, 0.1], iterations=1E3, 
                 ground_truth_y=None, cv_folds=5):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.regularization_strengths = regularization_strengths
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y
        self.cv_folds = cv_folds

    def fit(self, X):
        """Deconvolve data

        Args:
            X (2d array-like): first dimension is different measurements,
                               second dimension is intensities at different points
        """
        deconvolver = LRFisterDeconvolve(
            self.impulse_response_x,
            self.impulse_response_y,
            self.convolved_x,
            iterations=self.iterations,
            ground_truth_y=self.ground_truth_y,
            logging=False
        )
        param_grid = {'regularization_strength': self.regularization_strengths}
        if self.ground_truth_y is not None:
            scoring = {
                'deconvolved': deconvolution_metrics.neg_deconvolved_mse,
                'reconstruction': deconvolution_metrics.neg_reconstruction_mse}
        else:
            scoring = {'reconstruction': deconvolution_metrics.neg_reconstruction_mse}
        deconvolver_gs = GridSearchCV(
            deconvolver,
            param_grid,
            cv=self.cv_folds,
            return_train_score=True,
            verbose=True,
            scoring=scoring,
            refit='reconstruction',
            n_jobs=-1
        )
        deconvolver_gs.fit(X)
        self.best_regularization_strength_ = deconvolver_gs.best_params_
        self.measured_y_ = np.mean(X, axis=0)     # Average measurement
        self.deconvolved_y_ = deconvolver_gs.best_estimator_.deconvolved_y_
        self.reconstruction_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
        if self.ground_truth_y is not None:
            self.deconvolved_mse_ = -1*deconvolver_gs.cv_results_['mean_test_deconvolved']
            self.deconvolved_mse_std_ = deconvolver_gs.cv_results_['std_test_deconvolved']
        self.reconstruction_train_mse_ = -1*deconvolver_gs.cv_results_['mean_train_reconstruction']
        self.reconstruction_train_mse_std_ = deconvolver_gs.cv_results_['std_train_reconstruction']
        self.cv_ = -1*deconvolver_gs.cv_results_['mean_test_reconstruction']
        self.cv_std_ = deconvolver_gs.cv_results_['std_test_reconstruction']
        return self

    def predict(self, X=None):
        """Return estimated deconvolved signal.

        Args:
            X: unused, but included to be consistent with sklearn conventions
        """
        return self.deconvolved_y_

    def _get_deconvolved_x(self):
        """Return x-values (locations) of deconvolved signal.
        (photon energies for PAX)
        """
        average_impulse_x = np.mean(self.impulse_response_x)
        deconvolved_x = self.convolved_x-average_impulse_x
        return deconvolved_x


class LRDeconvolve(BaseEstimator):
    """Modified Lucy-Richardson deconvolution
    (the modification enables handling a background)
    """
    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, 
                 iterations=5, ground_truth_y=None, X_valid=None, 
                 logging=False):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y
        self.X_valid = X_valid
        self.logging = logging

    def fit(self, X):
        """Perform PAX 
        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        self.measured_y_ = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR(self.measured_y_)
        self.reconstruction_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
        return self

    def _LR(self, measured_y):
        """Perform modifed Lucy-Richardson deconvolution of measured_y
        """
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        if self.logging:
            writer = tf.summary.create_file_writer(f'{LOGDIR}unregularized')
        for iteration in range(self.iterations):
            ones_vec = np.ones_like(previous_O)
            blurred = convolve(self.impulse_response_y, previous_O, mode='valid')
            correction_factor = measured_y/blurred
            gradient = (convolve(impulse_response_y_reversed, ones_vec, mode='valid')
                        -convolve(impulse_response_y_reversed, correction_factor, mode='valid'))
            current_O = previous_O*(1-gradient)
            previous_O = current_O
            if self.logging:
                with writer.as_default():
                    self._save_iteration_stats(current_O, iteration)
            writer.flush()
        return current_O

    def _save_iteration_stats(self, current_deconvolved, iteration):
        """Calculate statistics from current result and save to tensorboard log
        """
        current_reconstruction = convolve(current_deconvolved, self.impulse_response_y, mode='valid')
        reconstruction_mse = mean_squared_error(current_reconstruction, self.measured_y_)
        tf.summary.scalar('train_reconstruction_mse', reconstruction_mse, step=iteration)
        if self.ground_truth_y is not None:
            # We have access to the ground truth, so we can calculate a few more metrics
            deconvolved_mse = mean_squared_error(current_deconvolved, self.ground_truth_y)
            tf.summary.scalar('deconvolved_mse', deconvolved_mse, step=iteration)
            ground_truth_reconstruction = convolve(self.ground_truth_y, self.impulse_response_y, mode='valid')
            reconstruction_mse = mean_squared_error(current_reconstruction, ground_truth_reconstruction)
            tf.summary.scalar('reconstruction_mse', reconstruction_mse, step=iteration)
        if self.X_valid is not None:
            val_reconstruction_mse = mean_squared_error(current_reconstruction, self.X_valid)
            tf.summary.scalar('validation_reconstruction_mse', val_reconstruction_mse, step=iteration)


    def _deconvolution_guess(self, measured_y):
        """Return initial guess for deconvolved signal
        (use blurred version of measured_y as guess)
        """
        sigma = 1
        x = self.impulse_response_x
        mu = np.mean(x)
        gauss = np.exp((-1/2)*((x-mu)/sigma)**2)
        gauss = gauss/np.sum(gauss)
        return convolve(measured_y, gauss, mode='valid')
        #return measured_y

    def predict(self, X=None):
        return self.deconvolved_y_
        
    def score(self, X_test):
        """Default scoring is reconstruction mean squared error
        """
        mean_X_test = np.mean(X_test, axis=0)
        mse = mean_squared_error(self.reconstruction_y_, mean_X_test)
        return -1*mse

    def _get_deconvolved_x(self):
        """Return x-values (locations) of deconvolved signal.
        (photon energies for PAX)
        """
        average_impulse_x = np.mean(self.impulse_response_x)
        deconvolved_x = self.convolved_x-average_impulse_x
        return deconvolved_x

class LRFisterDeconvolve(LRDeconvolve):
    """Modifed Lucy-Richardson deconvolution with Fister regularization
    (The modification enables handling of a background in the impulse response function)
    """

    def __init__(self, impulse_response_x, impulse_response_y, convolved_x, 
                 regularization_strength=0.05, iterations=1E2, ground_truth_y=None, 
                 X_valid=None, logging=False):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.reconstruction_x = convolved_x
        self.deconvolved_x = self._get_deconvolved_x()
        self.regularization_strength = regularization_strength
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y
        self.X_valid = X_valid
        self.logging = logging

    def _LR_fister(self, measured_y):
        """Perform Fister-regularized modified Lucy-Richardson deconvolution of measured_y
        """
        gauss = self._normalized_gaussian(
            self.impulse_response_x,
            np.mean(self.impulse_response_x),
            self.regularization_strength
        )
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        ones_vec = np.ones_like(previous_O)
        if self.logging:
            writer = tf.summary.create_file_writer(f'{LOGDIR}{self.regularization_strength}')
        for iteration in range(self.iterations):
            blurred = convolve(self.impulse_response_y, previous_O, mode='valid')
            correction_factor = measured_y/blurred
            gradient = (convolve(impulse_response_y_reversed, ones_vec, mode='valid')
                        -convolve(impulse_response_y_reversed, correction_factor, mode='valid'))
            current_O = previous_O*(1-gradient)
            current_O = convolve(current_O, gauss, mode='valid')
            previous_O = current_O
            if self.logging:
                with writer.as_default():
                    self._save_iteration_stats(current_O, iteration)
        return current_O

    def fit(self, X):
        """Deconvolve data
        parameters:
            X: array, rows are measurements, columns are points of the measurements
        """
        self.measured_y_ = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR_fister(self.measured_y_)
        self.reconstruction_y_ = convolve(self.deconvolved_y_, self.impulse_response_y, mode='valid')
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
