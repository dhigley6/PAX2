"""Classes for regularized scaled gradient deconvolution
These are equivalent to regularized Lucy-Richardson deconvolution in the case
that the impulse response function is negligible at the boundaries.
The algorithms are described in https://arxiv.org/abs/2006.10914

The regularization we employed here was originally suggested by Fister et al.
for regularizing Lucy-Richardson deconvolution. This consists of convolving
the estimate of the deconvolved signal with a Gaussian after each iteration
of deconvolution. The width of the Gaussian sets the strength of the regularizaion.
See the below reference for more details:
T. T. Fister et al. Phys. Rev. B 75, 174106 (2007)
"""

from typing import List, Optional
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve
from sklearn.model_selection import GridSearchCV

from pax_deconvolve.deconvolution import deconvolution_metrics

LOGDIR = "logdir/"  # Log directory to save tensorboard files in


class LRFisterGrid(BaseEstimator):
    """Fister-regularized deconvolution with regularization chosen by cross validation.

    Attributes:
        impulse_response_x {array-like of shape (n_i,)}: x-values (locations) 
            of impulse response
        impulse_response_y {array-like of shape (n_i,)}: y-values (intensities) 
            of impulse response
        convolved_x {array-like of shape (n_c,)}: x-values (locations) 
            of convolved data
        deconvolved_x {array-like of shape (n_c,)}: x-values (locations)
            of deconvolved data
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

    def __init__(
        self,
        impulse_response_x: np.ndarray,
        impulse_response_y: np.ndarray,
        convolved_x: np.ndarray,
        regularization_strengths: List[float] = [0.01, 0.1],
        iterations: int = 1e3,
        ground_truth_y: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.deconvolved_x = _get_deconvolved_x(
            self.convolved_x, self.impulse_response_x
        )
        self.regularization_strengths = regularization_strengths
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y
        self.cv_folds = cv_folds

    def fit(self, X: np.ndarray) -> "LRFisterGrid":
        """Deconvolve data

        Uses cross validation-like procedure to choose regularization strength

        Parameters:
            X (2d array-like): first dimension is different measurements,
                               second dimension is intensities at different points
        """
        self.measured_y_ = np.mean(X, axis=0)  # Average measurement
        cv_deconvolver = self._create_cv_deconvolution_estimator()
        cv_deconvolver.fit(X)
        self._record_deconvolution_results(cv_deconvolver)
        return self

    def _record_deconvolution_results(self, cv_deconvolver: GridSearchCV):
        """Record results of deconvolution as attributes of self
        """
        self.best_regularization_strength_ = cv_deconvolver.best_params_[
            "regularization_strength"
        ]
        self.deconvolved_y_ = cv_deconvolver.best_estimator_.deconvolved_y_
        self.reconstruction_y_ = convolve(
            self.deconvolved_y_, self.impulse_response_y, mode="valid"
        )
        self.reconstruction_train_mse_ = (
            -1 * cv_deconvolver.cv_results_["mean_train_reconstruction"]
        )
        self.reconstruction_train_mse_std_ = cv_deconvolver.cv_results_[
            "std_train_reconstruction"
        ]
        self.cv_ = -1 * cv_deconvolver.cv_results_["mean_test_reconstruction"]
        self.cv_std_ = cv_deconvolver.cv_results_["std_test_reconstruction"]
        if self.ground_truth_y is not None:
            # if the ground truth is known, we can calculate and record deconvolved errors
            self.deconvolved_mse_ = (
                -1 * cv_deconvolver.cv_results_["mean_test_deconvolved"]
            )
            self.deconvolved_mse_std_ = cv_deconvolver.cv_results_[
                "std_test_deconvolved"
            ]

    def _create_cv_deconvolution_estimator(self) -> GridSearchCV:
        """Create deconvolution estimator that uses cross validation for selecting regularization
        """
        deconvolver = LRFisterDeconvolve(
            self.impulse_response_x,
            self.impulse_response_y,
            self.convolved_x,
            iterations=self.iterations,
            ground_truth_y=self.ground_truth_y,
            logging=False,
        )
        param_grid = {"regularization_strength": self.regularization_strengths}
        scoring = {"reconstruction": deconvolution_metrics.neg_reconstruction_mse}
        if self.ground_truth_y is not None:
            # we have access to ground truth, so we can calculate deconvolved MSE
            scoring.update({"deconvolved": deconvolution_metrics.neg_deconvolved_mse})
        cv_deconvolver = GridSearchCV(
            deconvolver,
            param_grid,
            cv=self.cv_folds,
            return_train_score=True,
            verbose=True,
            scoring=scoring,
            refit="reconstruction",
            n_jobs=-1,
        )
        return cv_deconvolver

    def predict(self, X: None = None) -> np.ndarray:
        """Return estimated deconvolved signal.

        Parameters:
            X: unused, but included to be consistent with sklearn conventions
        """
        return self.deconvolved_y_


class LRDeconvolve(BaseEstimator):
    """Modified Lucy-Richardson deconvolution
    (the modification enables handling a background)

    Attributes:
        impulse_response_x {array-like of shape (n_i,)}: x-values (locations) 
            of impulse response
        impulse_response_y {array-like of shape (n_i,)}: y-values (intensities) 
            of impulse response
        convolved_x {array-like of shape (n_c,)}: x-values (locations) 
            of convolved data
        deconvolved_x {array-like of shape (n_c,)}: x-values (locations)
            of deconvolved data
        iterations (int): Number of iterations to use in deconvolution
        ground_truth_y {array-like of shape (n_c,)}: y-values (intensities) 
            of ground truth for deconvolution (None if not available)
        X_valid {array-like of shape (n_c,)}: mean of validation data
        logging (boolean): log data in tensorboard if True
        measured_y_ {array-like of shape (n_c,)}: Average measurement
        deconvolved_y_ {array-like of shape (n_c,)}: Deconvolved intensities
        reconstruction_y_ {array-like of shape (n_c,)}: Reconstruction of
            input data from deconvolved result
    """

    def __init__(
        self,
        impulse_response_x: np.ndarray,
        impulse_response_y: np.ndarray,
        convolved_x: np.ndarray,
        iterations: int = 5,
        ground_truth_y: Optional[np.ndarray] = None,
        X_valid: Optional[np.ndarray] = None,
        logging: bool = False,
    ):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.deconvolved_x = _get_deconvolved_x(
            self.convolved_x, self.impulse_response_x
        )
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y
        self.X_valid = X_valid
        self.logging = logging

    def fit(self, X: np.ndarray) -> "LRDeconvolve":
        """Perform deconvolution 

        parameters:
            X: array, rows are PAX spectra measurements, columns are specific electron energies
        """
        self.measured_y_ = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR(self.measured_y_)
        self.reconstruction_y_ = convolve(
            self.deconvolved_y_, self.impulse_response_y, mode="valid"
        )
        return self

    def _LR(self, measured_y: np.ndarray) -> np.ndarray:
        """Perform modifed Lucy-Richardson deconvolution of measured_y
        """
        previous_O = self._deconvolution_guess(measured_y)
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        if self.logging:
            writer = tf.summary.create_file_writer(f"{LOGDIR}unregularized")
        for iteration in range(self.iterations):
            ones_vec = np.ones_like(previous_O)
            blurred = convolve(self.impulse_response_y, previous_O, mode="valid")
            correction_factor = measured_y / blurred
            gradient = convolve(
                impulse_response_y_reversed, ones_vec, mode="valid"
            ) - convolve(impulse_response_y_reversed, correction_factor, mode="valid")
            current_O = previous_O * (1 - gradient)
            previous_O = current_O
            if self.logging:
                with writer.as_default():
                    self._save_iteration_stats(current_O, iteration)
            writer.flush()
        return current_O

    def _save_iteration_stats(self, current_deconvolved: np.ndarray, iteration: int):
        """Calculate statistics from current result and save to tensorboard log
        """
        current_reconstruction = convolve(
            current_deconvolved, self.impulse_response_y, mode="valid"
        )
        reconstruction_mse = mean_squared_error(
            current_reconstruction, self.measured_y_
        )
        tf.summary.scalar(
            "train_reconstruction_mse", reconstruction_mse, step=iteration
        )
        if self.ground_truth_y is not None:
            # We have access to the ground truth, so we can calculate a few more metrics
            deconvolved_mse = mean_squared_error(
                current_deconvolved, self.ground_truth_y
            )
            tf.summary.scalar("deconvolved_mse", deconvolved_mse, step=iteration)
            ground_truth_reconstruction = convolve(
                self.ground_truth_y, self.impulse_response_y, mode="valid"
            )
            reconstruction_mse = mean_squared_error(
                current_reconstruction, ground_truth_reconstruction
            )
            tf.summary.scalar("reconstruction_mse", reconstruction_mse, step=iteration)
        if self.X_valid is not None:
            val_reconstruction_mse = mean_squared_error(
                current_reconstruction, self.X_valid
            )
            tf.summary.scalar(
                "validation_reconstruction_mse", val_reconstruction_mse, step=iteration
            )

    def _deconvolution_guess(self, measured_y: np.ndarray) -> np.ndarray:
        """Return initial guess for deconvolved signal
        (use blurred version of measured_y with same length as deconvolved signal as guess)
        """
        sigma = 1  # width of Gaussian blur
        x = self.impulse_response_x
        mu = np.mean(x)
        gauss = _normalized_gaussian(x, mu, sigma)
        # gauss = np.exp((-1/2)*((x-mu)/sigma)**2)
        # gauss = gauss/np.sum(gauss)
        convolved = convolve(measured_y, gauss, mode="valid")
        num_pad = len(convolved) - len(self.deconvolved_x)
        return np.pad(
            convolved, pad_width=(0, num_pad), mode="constant", constant_values=0
        )

    def predict(self, X: None) -> np.ndarray:
        """Return estimated deconvolved signal.

        Parameters:
            X: unused, but included to be consistent with sklearn conventions
        """
        return self.deconvolved_y_

    def score(self, X_test: np.ndarray) -> float:
        """Default scoring is reconstruction mean squared error
        """
        mean_X_test = np.mean(X_test, axis=0)
        mse = mean_squared_error(self.reconstruction_y_, mean_X_test)
        return -1 * mse


class LRFisterDeconvolve(LRDeconvolve):
    """Modifed Lucy-Richardson deconvolution with Fister regularization
    (The modification enables handling of a background in the impulse response function)

    Attributes:
        impulse_response_x {array-like of shape (n_i,)}: x-values (locations) 
            of impulse response
        impulse_response_y {array-like of shape (n_i,)}: y-values (intensities) 
            of impulse response
        convolved_x {array-like of shape (n_c,)}: x-values (locations) 
            of convolved data
        regularization_strength {float}: regularization strength
        deconvolved_x {array-like of shape (n_c,)}: x-values (locations)
            of deconvolved data
        iterations (int): Number of iterations to use in deconvolution
        ground_truth_y {array-like of shape (n_c,)}: y-values (intensities) 
            of ground truth for deconvolution (None if not available)
        X_valid {array-like of shape (n_c,)}: mean of validation data
        logging (boolean): log data in tensorboard if True
        measured_y_ {array-like of shape (n_c,)}: Average measurement
        deconvolved_y_ {array-like of shape (n_c,)}: Deconvolved intensities
        reconstruction_y_ {array-like of shape (n_c,)}: Reconstruction of
            input data from deconvolved result
    """

    def __init__(
        self,
        impulse_response_x: np.ndarray,
        impulse_response_y: np.ndarray,
        convolved_x: np.ndarray,
        regularization_strength: float = 0.05,
        iterations: int = 1e2,
        ground_truth_y: Optional[np.ndarray] = None,
        X_valid: Optional[np.ndarray] = None,
        logging: bool = False,
    ):
        self.impulse_response_x = impulse_response_x
        self.impulse_response_y = impulse_response_y
        self.convolved_x = convolved_x
        self.deconvolved_x = _get_deconvolved_x(
            self.convolved_x, self.impulse_response_x
        )
        self.regularization_strength = regularization_strength
        self.iterations = int(iterations)
        self.ground_truth_y = ground_truth_y
        self.X_valid = X_valid
        self.logging = logging

    def _LR_fister(self, measured_y: np.ndarray) -> np.ndarray:
        """Perform Fister-regularized modified Lucy-Richardson deconvolution of measured_y
        """
        previous_O = self._deconvolution_guess(measured_y)  # initial guess
        regularization_gauss = self._calculate_regularizing_gaussian()
        impulse_response_y_reversed = np.flip(self.impulse_response_y)
        ones_vec = np.ones_like(measured_y)
        if self.logging:
            writer = tf.summary.create_file_writer(
                f"{LOGDIR}{self.regularization_strength}"
            )
        for iteration in range(self.iterations):
            blurred = convolve(self.impulse_response_y, previous_O, mode="valid")
            correction_factor = measured_y / blurred
            gradient = convolve(
                impulse_response_y_reversed, ones_vec, mode="valid"
            ) - convolve(impulse_response_y_reversed, correction_factor, mode="valid")
            current_O = previous_O * (1 - gradient)
            current_O = convolve(
                current_O, regularization_gauss, mode="same"
            )  # apply regularization
            previous_O = current_O
            if self.logging:
                with writer.as_default():
                    self._save_iteration_stats(current_O, iteration)
        return current_O

    def _calculate_regularizing_gaussian(self) -> np.ndarray:
        """Calculate Fister-style Gaussian for regularization
        """
        gauss = _normalized_gaussian(
            self.deconvolved_x,
            np.mean(self.deconvolved_x),
            self.regularization_strength,
        )
        return gauss

    def fit(self, X: np.ndarray) -> "LRFisterDeconvolve":
        """Deconvolve data

        parameters:
            X {array-like}: rows are measurements, columns are points of the measurements
        """
        self.measured_y_ = np.mean(X, axis=0)
        self.deconvolved_y_ = self._LR_fister(self.measured_y_)
        self.reconstruction_y_ = convolve(
            self.deconvolved_y_, self.impulse_response_y, mode="valid"
        )
        return self


def _normalized_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return a normalized gaussian function

    Parameters:
        x {array-like}: locations to calculate Gaussian atß
        mu: center of Gaussian
        sigma: standard deviation of Gaussian
    """
    norm_gauss = np.exp((-1 / 2) * ((x - mu) / sigma) ** 2)
    norm_gauss = norm_gauss / np.sum(norm_gauss)
    return norm_gauss


def _get_deconvolved_x(
    convolved_x: np.ndarray, impulse_response_x: np.ndarray
) -> np.ndarray:
    """Return x-values (locations) of deconvolved signal.
    (photon energies for PAX)
    """
    first_point = np.amin(convolved_x) - np.amax(impulse_response_x)
    spacing = np.abs(impulse_response_x[1] - impulse_response_x[0])
    impulse_len = len(impulse_response_x)
    convolved_len = len(convolved_x)
    deconvolved_len = impulse_len - convolved_len + 1
    deconvolved_x = np.arange(
        first_point, first_point + deconvolved_len * spacing, spacing
    )
    return deconvolved_x
