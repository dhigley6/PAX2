#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:03:51 2019
@author: dhigley
Calculate simulated PAX spectra with Poisson statistics.
"""

from typing import Tuple, Dict
import numpy as np
from scipy.signal import convolve

from pax_deconvolve.pax_simulations import model_rixs
from pax_deconvolve.pax_simulations import model_photoemission


def simulate_from_presets(
    total_log10_num_electrons: float,
    rixs: str,
    photoemission: str,
    num_simulations: int,
    energy_loss: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Simulate PAX spectra with input strings indicating model spectra to use

    Parameters:
    -----------
    total_log10_num_electrons: float
        Base 10 logarithm of the total number of electrons detected in
        PAX spectra
    rixs: str
        Which model rixs spectrum to use (see
        pax_deconvolve.pax_simulations.model_rixs.make_model_rixs)
    photoemission: str
        Which model photoemission spectrum to use (see
        pax_deconvolve.pax_simulations.model_photoemission.make_model_photoemission)
    num_simulations: int
        How many PAX spectra to simulate
    energy_loss: np.ndarray
        Energy loss values to simulate (eV)

    Returns:
    --------
    impulse_response_xy: dictionary
        impulse response; 'x' key gives locations (1d), 'y' key gives
        intensities (1d)
    pax_xy: dictionary
        pax spectra; 'x' key gives electron kinetic energies (1d), 'y' key
        gives intensities (2d). Each row of intensities is a different
        simulated PAX spectrum, and each column is a different electron
        kinetic energy.
    xray_xy: dictionary
        xray spectrum; 'x' key gives photon energies (1d), 'y' key gives
        intensities (1d)
    """
    total_counts = 10 ** total_log10_num_electrons
    xray_xy = model_rixs.make_model_rixs(rixs, energy_loss)
    photoemission_xy = model_photoemission.make_model_photoemission(
        photoemission, xray_xy["x"]
    )
    impulse_response_xy = calculate_pax_impulse_response(
        photoemission_xy["x"], photoemission_xy["y"]
    )
    pax_y = simulate(
        xray_xy["y"], impulse_response_xy["y"], total_counts, num_simulations
    )
    pax_xy = {
        "x": _calculate_pax_kinetic_energy(xray_xy["x"], photoemission_xy["x"]),
        "y": pax_y,
    }
    return impulse_response_xy, pax_xy, xray_xy


def simulate(
    xray_y: np.ndarray,
    impulse_response_y: np.ndarray,
    counts: int,
    num_simulations: int = 1,
) -> np.ndarray:
    """Simulate PAX spectra from input xray and impulse responses

    Parameters:
    -----------
    xray_y: np.ndarray
        Intensities of X-ray spectrum
    impulse_response_y: np.ndarray
        Intensities of impulse response
    counts: int
        Total number of counts (detected electrons) in all PAX spectra
    num_simulations: int
        Number of PAX spectra to simulate

    Returns:
    --------
    pax_y: np.ndarray
        2d array of PAX intensities. Each row is a different PAX spectrum,
        and each column is a different electron kinetic energy.
    """
    noiseless_pax_spectrum = convolve(xray_y, impulse_response_y, mode="valid")
    single_electron = num_simulations * np.sum(noiseless_pax_spectrum) / counts
    pax_y = np.array(
        [
            _apply_poisson_noise(noiseless_pax_spectrum, single_electron)
            for _ in range(num_simulations)
        ]
    )
    return pax_y


def calculate_pax_impulse_response(
    photoemission_x: np.ndarray, photoemission_y: np.ndarray
) -> Dict[str, np.ndarray]:
    """Normalize and flip photoemission to obtain PAX impulse response.

    Parameters:
    -----------
    photoemission_x: np.ndarray
        Binding energies (eV) of photoemission spectrum
    photoemission_y: np.ndarray
        Intensities of photoemission spectrum

    Returns:
    --------
    impulse_response: dictionary
        Impulse response function for PAX. 'x' key gives locations, 'y' key
        gives intensities
    """
    impulse_response = {
        "x": -1 * photoemission_x,
        "y": np.flipud(photoemission_y),
    }
    norm_factor = np.sum(impulse_response["y"])
    impulse_response["y"] = impulse_response["y"] / norm_factor
    return impulse_response


def _apply_poisson_noise(data: np.ndarray, single_electron: float = 1.0) -> np.ndarray:
    """Apply Poisson noise to input data

    single_photon is the number of counts that corresponds to a single
    detected photon.

    Parameters:
    -----------
    data: np.ndarray
        Noiseless data to apply Poisson noise to
    single_electron: float
        Integrated intensity for a single electron (electron)

    Returns:
    --------
    output: np.ndarray
        Data with applied Poisson noise
    """
    output = np.random.poisson(data / single_electron) * single_electron
    return output


def _calculate_pax_kinetic_energy(
    xray_x: np.ndarray, impulse_response_x: np.ndarray
) -> np.ndarray:
    """Calculate electron kinetic energies of PAX spectrum

    Parameters:
    -----------
    xray_x: np.ndarray
        Photon energies (eV) of X-ray spectrum
    impulse_response_x: np.ndarray
        X-values (eV) of PAX impulse response function

    Returns:
    --------
    kinetic_energy: np.ndarray
        Kinetic energies (eV) of electrons in PAX spectrum
    """
    first_point = xray_x[0] - impulse_response_x[0]
    spacing = xray_x[1] - xray_x[0]
    pax_length = len(impulse_response_x) - len(xray_x) + 1
    kinetic_energy = np.arange(first_point, first_point + pax_length * spacing, spacing)
    return kinetic_energy
