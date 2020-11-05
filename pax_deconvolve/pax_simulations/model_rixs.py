#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:52:26 2018
@author: dhigley
Generate model RIXS spectra given input X-ray energy losses.
"""

from typing import Dict, Union, Callable, List
import numpy as np

INCIDENT_ENERGY = 778.0  # Default incident photon energy


def make_model_rixs(
    rixs: Union[str, List[Union[int, str]]],
    energy_loss: np.ndarray,
    incident_photon_energy: float = INCIDENT_ENERGY,
) -> Dict[str, np.ndarray]:
    """Return model RIXS photon energies and intensities

    Parameters:
    -----------
    rixs: str or List
        Which model RIXS function to use. Must be one of
            - 'schlappa'
            - 'georgi'
            - ['doublet', separation]
            - ['i_doublet', separation]
        In the last two cases, the second entry of the list is the separation of the doublet peaks
    energy_loss: (N,) array_like
        One-dimensional energy loss values (units of eV)
    incident_photon_energy: float
        Incident photon energy to use for X-rays (eV)

    Returns:
    --------
    rixs_xy: dictionary
        'x' key gives photon energies of RIXS spectrum
        'y' key gives intensities of RIXS spectrum
    """
    rixs_xy = _model_rixs_function(rixs)(energy_loss, incident_photon_energy)
    return rixs_xy


def _model_rixs_function(
    rixs: Union[str, List[Union[int, str]]]
) -> Callable[..., Dict[str, np.ndarray]]:
    """Return function for generating model RIXS spectrum

    Parameters:
    -----------
    rixs: str
        Which model RIXS function to use. Must be one of
            - 'schlappa'
            - 'georgi'
            - ['doublet', separation]
            - ['i_doublet', separation]
        In the last two cases, the second entry of the list is the separation of the doublet peaks

    Returns:
    --------
    out: function
        Model RIXS function
    """
    if rixs == "schlappa":
        return get_schlappa_rixs
    elif rixs == "georgi":
        return get_georgi_rixs
    elif isinstance(rixs, list):
        if rixs[0] == "doublet":
            separation = rixs[1]
            doublet = lambda x, incident_photon_energy: get_doublet(
                x, incident_photon_energy, separation=separation
            )
            return doublet
        elif rixs[0] == "i_doublet":
            separation = rixs[1]
            c_doublet = lambda x, incident_photon_energy: get_independent_doublet(
                x, incident_photon_energy, separation=separation
            )
            return c_doublet
        else:
            raise ValueError('Invalid "rixs" type')
    else:
        raise ValueError('Invalid "rixs" type')


def get_doublet(
    energy_loss: np.ndarray,
    incident_energy: float = INCIDENT_ENERGY,
    separation: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Doublet with seperation = 10xwidth

    Parameters:
    -----------
    energy_loss: (N,) array_like
        One-dimensional energy loss values (units of eV)
    incident_energy: float, optional
        Incident photon energy to use for X-rays (eV)
    separation: float, optional
        Peak-to-peak separation of doublet (eV)

    Returns:
    --------
    doublet: dictionary
        'x' key gives photon energies
        'y' key gives intensities
    """
    width = separation / 10
    elastic_peak = np.exp(-(((energy_loss - 0) / width) ** 2))
    loss_peak = np.exp(-(((energy_loss - separation) / width) ** 2))
    y = elastic_peak + loss_peak
    y = y / np.sum(y)
    doublet = {"x": incident_energy - np.flipud(energy_loss), "y": np.flipud(y)}
    return doublet


def get_independent_doublet(
    energy_loss: np.ndarray,
    incident_energy: float = INCIDENT_ENERGY,
    separation: float = 0.5,
    fwhm: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Doublet with independent separation and peak width

    Parameters:
    -----------
    energy_loss: (N,) array_like
        One-dimensional energy loss values (units of eV)
    incident_energy: float, optional
        Incident photon energy to use for X-rays (eV)
    separation: float, optional
        Peak-to-peak separation of doublet (eV)
    fwhm: float, optional
        Full-width-at-half-maximum of each peak of the doublet (eV)

    Returns:
    --------
    doublet: dictionary
        'x' key gives photon energies
        'y' key gives intensities
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    elastic_peak = np.exp(-(((energy_loss - 0) / sigma) ** 2))
    loss_peak = np.exp(-(((energy_loss - separation) / sigma) ** 2))
    y = elastic_peak + loss_peak
    y = y / np.sum(y)
    doublet = {"x": incident_energy - np.flipud(energy_loss), "y": np.flipud(y)}
    return doublet


def get_georgi_rixs(
    energy_loss: np.ndarray, incident_energy: float = INCIDENT_ENERGY
) -> Dict[str, np.ndarray]:
    """Return RIXS spectrum made up by Georgi

    Parameters:
    -----------
    energy_loss: (N,) array_like
        One-dimensional energy loss values (units of eV)
    incident_energy: float, optional
        Incident photon energy to use for X-rays (eV)

    Returns:
    --------
    georgi_rixs: dictionary
        'x' key gives photon energies
        'y' key gives intensities
    """
    p1 = np.exp(-(((energy_loss - 0) / 0.02) ** 2))
    p2 = 6 * np.exp(-(((energy_loss - 0.1) / 0.02) ** 2))
    p3 = 1.5 * np.exp(-(((energy_loss - 0.2) / 0.05) ** 2))
    p4 = 2 * np.exp(-(((energy_loss - 0.5) / 0.03) ** 2))
    p5 = 1.75 * np.exp(-(((energy_loss - 0.6) / 0.1) ** 2))
    p6 = 1.5 * np.exp(-(((energy_loss - 0.9) / 0.1) ** 2))
    y = p1 + p2 + p3 + p4 + p5 + p6
    georgi_rixs = {"x": incident_energy - np.flipud(energy_loss), "y": np.flipud(y)}
    return georgi_rixs


def get_schlappa_rixs(
    energy_loss: np.ndarray, incident_energy: float = INCIDENT_ENERGY
) -> Dict[str, np.ndarray]:
    """Return RIXS spectrum to approximate that measured in Schlappa et al.

    Parameters were chosen to approximate the spectrum shown in Fig. 2b of
    J. Schlappa et al., "Spin-orbital separation in the quasi-one-dimensional
    Mott insulator Sr_2CuO_3"
    Nature 485, 82-85 (2012)


    Parameters:
    -----------
    energy_loss: (N,) array_like
        One-dimensional energy loss values (units of eV)
    incident_energy: float, optional
        Incident photon energy to use for X-rays (eV)

    Returns:
    --------
    schlappa_rixs: dictionary
        'x' key gives photon energies
        'y' key gives intensities
    """
    p1 = 8 * np.exp(-(((energy_loss - 0.25) / 0.05) ** 2))
    p2 = 23 * np.exp(-(((energy_loss - 1.8) / 0.2) ** 2))
    p3 = 26 * np.exp(-(((energy_loss - 2.2) / 0.2) ** 2))
    p4 = 9 * np.exp(-(((energy_loss - 2.9) / 0.3) ** 2))
    p5 = 3 * np.exp(-(((energy_loss - 4.5) / 0.5) ** 2))
    p6 = 3 * np.exp(-(((energy_loss - 5.2) / 0.75) ** 2))
    y = p1 + p2 + p3 + p4 + p5 + p6
    y = np.flipud(y)
    y = np.clip(y, 0, None)
    schlappa_rixs = {"x": incident_energy - np.flipud(energy_loss), "y": y}
    return schlappa_rixs
