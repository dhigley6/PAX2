# Deconvolution of PAX data

This repository contains code for deconvolving PAX data as well as simulating PAX data, as described in detai in [1].

## Overview of PAX

Standard X-ray spectra estimations are accomplished by measuring X-rays directly with a grating-based spectrometer. Unfortunately, these grating-based spectrometers can be very large (up to hundreds of m$^2$ for measurements with state of the art resolutions) and have very low throughputs (~1 measured photon for every 10$^8$ photons absorbed in a sample is typical). PAX is alternative method for estimating X-ray spectra that could mitigate these challenges in many circumstances, serving as a complement to grating-based spectrometers. PAX transforms the X-ray spectra measurement problem to an electron spectra measurement problem, enabling use of electron spectrometers instead of X-ray spectrometers for X-ray spectroscopy. Electron spectrometers can be much smaller and have higher collection efficiencies than grating spectrometers with similar resolutions. 

In PAX, X-rays to be measured are incident on a converter system, where absorption of the X-rays generates photoelectrons. The emitted photoelectrons are then detected with a photoelectron spectrometer, giving the PAX spectrum. The PAX spectrum is approximately the convolution of the photoemission spectrum of the converter material when exposed to single color (monochromatic) X-ray radiation and the X-ray spectrum that we want to estimate. This photoemission spectrum can be easily measured with high accuracy. A common such photoemission spectrum consists of two narrow peaks and a non-uniform background. The desired X-ray spectrum is estimated from the measured PAX and photoemission spectra through deconvolution.

While inferring information about X-ray spectra in this manner is an idea that was first proposed in the 60s, the PAX method has not yet seen significant adoption. A main reason for this was concerns about whether X-ray spectra could be estimated robustly with PAX. In [1], we proposed an algorithm for analyzing PAX data and showed that X-ray spectra could be robustly estimated from previously recorded PAX data moderate resolution (~0.5 eV) and counts (~10$^4$) [2]. Further simulations [1] showed the promise of PAX for estimation of fine features (~0.1 eV scale) of X-ray spectra under experimental conditions that we think are achievable. This repository has code for the algorithm used in [1].

## Overview of Algorithm

We assume we can model the expected value of a measured PAX spectrum as a discrete convolution over a certain range (Eq. 4),

?

It is conveinent to write this as 

## Installation

```
pip install git+https://github.com/dhigley6/PAX2
```

## Usage



## Examples

See [Demonstration Notebook](https://github.com/dhigley6/PAX2/blob/master/notebooks/demonstration.ipynb) in notebooks folder. The notebook can also be accessed through Google Colaboratory:
[Demonstration Notebook](https://colab.research.google.com/github/dhigley6/PAX2/blob/master/notebooks/demonstration.ipynb)

## References

[1] D. J. Higley, H. Ogasawara, S. Zohar and G. L. Dakovski, "Using Photoelectron Spectroscopy to Measure Resonant Inelastic X-Ray Scattering: A Computational Investigation" (under review, arxiv version: https://arxiv.org/abs/2006.10914)
[2] 
