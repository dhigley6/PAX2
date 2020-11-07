# Deconvolution of PAX data

This repository contains code for deconvolving PAX data as well as simulating PAX data, as described in detail in [1].

## Overview of PAX

Resonant Inelastic X-ray Scattering (RIXS) is an increasingly widely utilized technique for studying elementary excitations in a wide range of matter. In RIXS, one directs monochromatic X-ray radiation onto matter under study and measures the spectra (energy distribution) of resultant scattered X-rays. Standard RIXS estimations are accomplished by measuring X-rays directly with a grating-based spectrometer. Unfortunately, these grating-based spectrometers can be very large (up to hundreds of m<sup>2</sup> for measurements with state of the art resolutions) and have very low throughputs (~1 measured photon for every 10<sup>8</sup> photons absorbed in a sample is typical). PAX is an alternative method for estimating X-ray spectra that could mitigate these challenges in many circumstances, serving as a complement to grating-based spectrometers. PAX transforms the X-ray spectra measurement problem to an electron spectra measurement problem, enabling use of electron spectrometers instead of X-ray spectrometers for RIXS. Electron spectrometers can be much smaller and have higher collection efficiencies than grating spectrometers with similar resolutions. 

In PAX, X-rays to be measured are incident on a converter material (or other system), where absorption of the X-rays generates photoelectrons. The emitted photoelectrons are then detected with a photoelectron spectrometer, giving the PAX spectrum. The PAX spectrum is approximately the convolution of the photoemission spectrum of the converter material when exposed to single color (monochromatic) X-ray radiation and the X-ray spectrum that we want to estimate. The photoemission spectrum can be easily measured with high accuracy. A common such spectrum consists of two narrow peaks and a non-uniform background. The desired X-ray spectrum is estimated from the measured PAX and photoemission spectra through deconvolution.

While inferring information about X-ray spectra in this manner is an idea that was first proposed in the 1960s, the PAX method has not yet seen significant adoption. A main reason for this was concerns about whether X-ray spectra could be estimated robustly with PAX. In [1], we proposed a deconvolution algorithm for analyzing PAX data and showed that X-ray spectra could be robustly estimated from previously recorded PAX data moderate resolution (~0.5 eV) and counts (~10$^4$) [2]. Further simulations [1] showed the promise of PAX for estimation of fine features (~0.1 eV scale) of X-ray spectra under experimental conditions that we think are achievable. This repository has code for the algorithm used in [1].

## Overview of Algorithm

(See [1] for a detailed description).

We assume we can model measured PAX spectra as a Poisson process with an expected value that is the discrete convolution over a certain range of the converter material photoemssion spectrum and the desired X-ray spectrum. From this, we derived a scaled gradient iteration for maximizing the likelihood of the estimated RIXS spectra given the measured data. This maximum likelihood estimatation is regularized by convolving with a Gaussian after each iteration, as originally proposed in [3] for deconvolution of different types of data. The regularization strength is set by the width of this Gaussian, with wider Gaussians enforcing higher degrees of smoothness of the estimated X-ray spectra. We determine a criterion for estimating the optimal regularization strength in a self-supervised manner. We take the estimated RIXS spectrum, convolve it with the measured photoemission spectrum and determine the mean squared difference of this with respect to a different PAX spectrum measured in idential conditions. We showed with simulations in [1] that minimizing this mean squared difference approximately gives the regularization strength where the mean squared difference of estimated RIXS spectra and true RIXS spectra is minimized.

## Installation

```
pip install git+https://github.com/dhigley6/PAX2
```

## Usage

All of the functionality that is intended for use is described below. The central object is the LRFisterGrid class. This is a [Scikit-Learn](scikit-learn.org) estimator-style class which estimates an X-ray spectrum given a set of PAX spectra measured under approximately identical conditions and the measured converter material photoemission spectrum. The best regularization strength is estimated from the data before estimating the X-ray spectrum. If one already knows the regularization strength they want to use, then they can use the LRFisterDeconvolve class.

Before running analysis, import the package,

```
import pax_deconvolve
```

### Estimating X-ray spectra from PAX data

X-ray spectra can be estimated from PAX data using the LRFisterGrid class. This is done by initializing the class, then running the fit method on PAX data. The result can be accessed for further processing with the predict method.

```
deconvolver = pax_deconvolve.LRFisterGrid(
    impulse_response_x,
    impulse_response_y,
    convolved_x,
    regularization_strengths,
    iterations,
    cv_folds=cv_folds,
    ground_truth_y=ground_truth_y,
)
_ = deconvolver.fit(pax_spectra_y)
estimated_best_regularization_strength = deconvolver.best_regularization_strength_
estimated_xray_y = deconvolver.predict()
estimated_xray_x = deconvolver.deconvolved_x
```
The input variables are

- impulse_response_x: (n_i,)-shaped array with negative one times the binding energy of the converter material photoemission spectrum (ordered from lowest to highest)
- impulse_response_y: (n_i,)-shaped array with the intensity of the converter material photoemission spectrum and the same indicies as impulse_response_x
- convolved_x: (n_c,)-shaped array of the electron kinetic energies of the PAX spectra
- regularization_strengths: List of regularization strengths to optimize over
- iterations: Number of iterations to run deconvolution for
- pax_spectra_y: (m, n_c)-shaped array with the measured PAX spectra (Each row is a PAX spectrum measured under identical conditions. As currently implemented, this requires at least two PAX spectra to work.)
- cv_folds (not required, default = 5): Number of cross validation folds to use in estimating the optimal regularization strength
- ground_truth_y (not required): (n_x,)-shaped ground truth X-ray spectrum, if known. Providing this will enable more visualizations (see below), but does not affect the deconvolution.

The output variables are

- estimated_best_regularization_strength: The estimated best regularization strength
- estimated_xray_x: (n_x,)-shaped array of the photon energies of the estimated X-ray spectra
- estimated_xray_y: (n_x,)-shaped array of the intensities of the estimated X-ray spectra

### Estimating X-ray spectra from PAX data with Known Regularization Strength

If one already knows the regularization strength they want to use, then the LRFisterDeconvolve class can be used as follows

```
deconvolver = pax_deconvolve.LRFisterDeconvolve(
    impulse_response_x,
    impulse_response_y,
    convolved_x,
    regularization_strengths,
    iterations,
    cv_folds=cv_folds,
    ground_truth_y=ground_truth_y,
)
_ = deconvolver.fit(pax_spectra_y)
estimated_xray_y = deconvolver.predict()
estimated_xray_x = deconvolver.deconvolved_x
```

The variables have the same definitions as for LRFisterGrid.

### Assessing Convergence

One can assess whether a certain number of iterations are sufficient by running the following code in a Jupyter notebook:

```
pax_deconvolve.assess_convergence(
    impulse_response_x,
    impulse_response_y,
    pax_spectra_x,
    pax_spectra_y,
    regularization_strengths,
    iterations,
)
%tensorboard --logdir logdir
```

This will open some interactive plots which one can use to assess whether enough iterations have been run, as described in [1]. The variables have the same definitions as above.

### Visualizing results

The package includes some functions for plotting deconvolution results for conveinence. plot_photoemission and plot_result can be run on fitted instances of LRFisterGrid or LRDeconvolve while plot_cv only runs on instances of LRDeconvolve.

```
pax_deconvolve.plot_photoemission(deconvolver)
pax_deconvolve.plot_result(deconvolver)
pax_deconvolve.plot_cv(deconvolver)
```

(deconvolver is an instance where the fit method has already been run.)

### Simulating PAX data

The package includes a function for simulating PAX data under certain conditions. It is recommended to use this with the 'schlappa' model RIXS spectrum, which (very roughly) models a RIXS spectrum from a paper by Justine Schlappa and others.

```
impulse_response_xy, pax_spectra_xy, xray_xy = pax_deconvolve.simulate_from_presets(
    log10_num_electrons,
    'schlappa',
    photoemission_mode,
    num_simulations,
    energy_loss,
)
impulse_response_x, impulse_response_y = impulse_response_xy['x'], impulse_response_xy['y']
pax_spectra_x, pax_spectra_y = pax_spectra_xy['x'], pax_spectra_xy['y']
xray_x, xray_y = xray_xy['x'], xray_xy['y']
```

The output variables have the same meanings as defined above. The inputs are defined as

- log10_num_electrons: Base 10 logarithm of the integrated number of detected electrons in the simulated PAX spectra
- photoemission_model: Which photoemission model to use, must be one of
  - 'ag'
  - 'ag_with_bg'
  - 'au_4f'
- num_simulations: Number of PAX spectra to simulate
- energy_loss: Energy loss values for X-ray spectrum

## Examples

Example showing deconvolution of moderate resolution (~0.5 eV) data from [2]: [Experiment](https://github.com/dhigley6/PAX2/blob/master/notebooks/tutorial_on_lcls_data.ipynb).

Example showing deconvolution of higher resolution simulated data: [Simulated](https://github.com/dhigley6/PAX2/blob/master/notebooks/demonstration.ipynb). The notebook can also be accessed through Google Colaboratory:
[Demonstration Notebook](https://colab.research.google.com/github/dhigley6/PAX2/blob/master/notebooks/demonstration.ipynb).

## References

[1] D. J. Higley, H. Ogasawara, S. Zohar and G. L. Dakovski, "Using Photoelectron Spectroscopy to Measure Resonant Inelastic X-Ray Scattering: A Computational Investigation" (under review, arxiv version: https://arxiv.org/abs/2006.10914).

[2] G. L. Dakovski, M.-F. Lin, D. S. Damiani, W. F. Schlotter, J. J. Turner, D. Nordlund, and H. Ogasawara, J. Synchrotron Radiat. **24**, 1180 (2017).

[3] T. T. Fister, G. T. Seidler, J. J. Rehr, J. J. Kas, W. T. Elam, J. O. Cross, and K. P. Nagle, Phys. Rev. B **75**, 174106 (2007).
