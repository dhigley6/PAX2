#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:12:02 2018
@author: dhigley
Generate model photoemission spectra given input binding energies.
"""

from typing import Dict, Callable
import numpy as np

BOLTZMANN_CONSTANT = 8.617e-5  # (eV/K) Taken from wikipedia

AU_4F_7HALF_BINDING = 84.0  # Taken from X-ray data booklet
AU_4F_5HALF_BINDING = 87.6  # Taken from X-ray data booklet
AU_4F_BROAD = 0.335  # Lorentzian FWHM taken from Y. Takata et al.,
# Nuclear Instruments and Methods in Physics
# Research A 547 (2005) 50-55, "Development of
# hard X-ray photoelectron spectroscopy at BL29XU
# in SPring-8
AU_4F_CENTER = 85.8  # ~Center of Au 4f photoemission

AG_3D_5HALF_BINDING = 368.3  # Taken from X-ray data booklet
AG_3D_3HALF_BINDING = 374.0  # Taken from X-ray data booklet
AG_3D_BROAD = 0.233  # Lorentzian intrinsic broadening taken from G. Panaccione
# et al., "High-energy photoemission in silver: resolving
# d and sp contributions in valence band spectra" J.
# Phys. Condens. Matter 17 (2005) 2671-2679
AG_3D_CENTER = 372

FERMI_CENTER = 0


def make_model_photoemission(
    photoemission: str, xray_x: np.ndarray
) -> Dict[str, np.ndarray]:
    """Return x and y values for specified photoemission function

    Parameters:
    -----------
    photoemission: str
        name of model photoemission function to use. Should be one of
            - 'ag'
            - 'ag_with_bg'
            - 'fermi'
            - 'au_4f'
    xray_x: (N,) array_like
        One-dimensional x-values (locations) of X-ray spectrum

    Returns:
    --------
    photoemission_xy: dictionary
        dictionary of model photoemission spectrum. 'x' key gives locations, 'y' key gives intensities
    """
    if photoemission == "ag":
        center = AG_3D_CENTER
    elif photoemission == "ag_with_bg":
        center = AG_3D_CENTER
    elif photoemission == "fermi":
        center = FERMI_CENTER
    elif photoemission == "au_4f":
        center = AU_4F_CENTER
    energy_spacing = xray_x[1] - xray_x[0]
    binding_energy = calculate_binding_energies(len(xray_x), energy_spacing, center)
    photoemission_xy = _model_photoemission_function(photoemission)(binding_energy)
    return photoemission_xy


def _model_photoemission_function(
    photoemission: str,
) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    """Return function to make model photoemission spectrum

    Parameters:
    -----------
    photoemission: str
        name of model photoemission function to use. Should be one of
            - 'ag'
            - 'ag_with_bg'
            - 'fermi'
            - 'au_4f'

    Returns:
    --------
    out: function
        model photoemission function
    """
    if photoemission == "ag":
        return get_ag_3d_spectrum
    elif photoemission == "ag_with_bg":
        return get_ag_3d_with_bg
    elif photoemission == "fermi":
        return get_fermi_dirac
    elif photoemission == "au_4f":
        return get_au_4f_spectrum
    else:
        raise ValueError('Invalid "photoemission" type')


def calculate_binding_energies(
    points_in_spectrum: int, energy_spacing: float, center_binding_energy: float
) -> np.ndarray:
    """Calculate appropriate binding energies to use for simulation

    The binding energies should be
      - centered around the main photoemission features,
      - be length 2xlen(xray_spectrum)-1
      - have the same energy spacing as the X-ray spectrum

    Parameters:
    -----------
    points_in_spectrum: int
        Number of points in binding energy
    energy_spacing: float
        Energy separation between bindng energies
    center_binding_energy: float
        Center of binding energies to return

    Returns:
    --------
    binding_energies: (N,) array_like
        Binding energies of photoemission spectrum
    """
    binding_energies_len = 2 * points_in_spectrum - 1
    binding_energies = np.arange(
        0, binding_energies_len * energy_spacing - energy_spacing / 2, energy_spacing
    )
    binding_energies = (
        binding_energies - np.mean(binding_energies) + center_binding_energy
    )
    return binding_energies


def get_au_4f_spectrum(binding_energy: np.ndarray) -> Dict[str, np.ndarray]:
    """Return model photoemission spectrum for Au 4f levels

    Parameters:
    -----------
        binding_energy: (N,) array_like
            Binding energies of photoemission spectrum

    Returns:
    --------
        au_4f_photoemission: dictionary
            'x' key gives binding energies
            'y' key gives intensities
    """
    b2 = AU_4F_BROAD / 2
    au_4f_7half = b2 / ((binding_energy - AU_4F_7HALF_BINDING) ** 2 + (b2) ** 2)
    au_4f_5half = b2 / ((binding_energy - AU_4F_5HALF_BINDING) ** 2 + (b2) ** 2)
    au_4f_background = 0.15
    au_4f_photoemission = {
        "x": binding_energy,
        "y": 3 * au_4f_7half + 4 * au_4f_5half + au_4f_background,
    }
    return au_4f_photoemission


def get_ag_3d_with_bg(binding_energy: np.ndarray) -> Dict[str, np.ndarray]:
    """Return photoemission spectrom for Ag 3d levels with artificial background added

    Parameters:
    -----------
        binding_energy: (N,) array_like
            Binding energies of photoemission spectrum

    Returns:
    --------
        ag_3d_spectrum_with_bg: dictionary
            'x' key gives binding energies
            'y' key gives intensities
    """
    raw_ag_3d_spectrum = get_ag_3d_spectrum(binding_energy)
    bg_height = np.amax(raw_ag_3d_spectrum["y"]) / 10
    bg = bg_height / (1 + np.exp(-1 * (raw_ag_3d_spectrum["x"] - AG_3D_5HALF_BINDING)))
    ag_3d_spectrum_with_bg = raw_ag_3d_spectrum
    ag_3d_spectrum_with_bg["y"] = raw_ag_3d_spectrum["y"] + bg
    return ag_3d_spectrum_with_bg


def get_ag_3d_spectrum(binding_energy: np.ndarray) -> Dict[str, np.ndarray]:
    """Return model photoemission spectrum for Ag 3d levels

    Parameters:
    -----------
        binding_energy: (N,) array_like
            Binding energies of photoemission spectrum

    Returns:
    --------
        ag_3d_photoemission: dictionary
            'x' key gives binding energies
            'y' key gives intensities
    """
    b2 = AG_3D_BROAD / 2  # abbreviation for half of broadening
    ag_3d_5half = (b2) / ((binding_energy - AG_3D_5HALF_BINDING) ** 2 + (b2) ** 2)
    ag_3d_3half = (b2) / ((binding_energy - AG_3D_3HALF_BINDING) ** 2 + (b2) ** 2)
    ag_3d_background = 0
    y = 5 * ag_3d_5half + 3 * ag_3d_3half + ag_3d_background
    y = y / np.sum(y)
    ag_3d_photoemission = {"x": binding_energy, "y": y}
    return ag_3d_photoemission


def get_fermi_dirac(binding_energy: np.ndarray, T: float = 4) -> Dict[str, np.ndarray]:
    """Model Fermi edge spectrum

    Parameters:
    -----------
        binding_energy: (N,) array_like
            Binding energies of photoemission spectrum
        T: float
            Temperature of Fermi-Dirac distribution

    Returns:
    --------
        au_4f_photoemission: dictionary
            'x' key gives binding energies
            'y' key gives intensities
    """
    kbt = T * BOLTZMANN_CONSTANT
    y = 1 / (np.exp(-1 * binding_energy / kbt) + 1)
    fermi_dirac_photoemission = {
        "x": binding_energy,
        "y": y / np.sum(y) + 1e-9,
    }
    return fermi_dirac_photoemission
