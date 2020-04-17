#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:52:26 2018
@author: dhigley
Generate model RIXS spectra given input X-ray energy losses.
"""

import numpy as np

INCIDENT_ENERGY = 778    # Default incident photon energy

def make_model_rixs(rixs, energy_spacing, incident_photon_energy=INCIDENT_ENERGY):
    #energy_loss = np.arange(-0.5, 0.5, energy_spacing)
    energy_loss = np.arange(-8, 10, energy_spacing)
    rixs_xy = _model_rixs_function(rixs)(
        energy_loss,
        incident_photon_energy)
    return rixs_xy

def _model_rixs_function(rixs):
    if rixs == 'schlappa':
        return get_schlappa_rixs
    elif rixs == 'georgi':
        return get_georgi_rixs
    elif isinstance(rixs, list):
        if rixs[0] == 'doublet':
            separation = rixs[1]
            doublet = lambda x, incident_photon_energy: get_doublet(
                x,
                incident_photon_energy,
                separation=separation)
            return doublet
        else:
            raise ValueError('Invalid "rixs" type')
    else:
        raise ValueError('Invalid "rixs" type')

def get_elastic_peak(energy_loss, incident_energy=INCIDENT_ENERGY):
    """RIXS spectrum consisting of a single peak at zero energy loss
    """
    y = 6*np.exp(-((energy_loss-0)/0.02)**2)
    elastic_peak = {'x': incident_energy-np.flipud(energy_loss),
                    'y': np.flipud(y),
                    'x_min': -0.2+incident_energy,
                    'x_max': 0.2+incident_energy}
    return elastic_peak

def get_doublet(energy_loss, incident_energy=INCIDENT_ENERGY, separation=0.5):
    """Doublet with seperation = 10xwidth
    """
    width = separation/10
    elastic_peak = np.exp(-((energy_loss-0)/width)**2)
    loss_peak = np.exp(-((energy_loss-separation)/width)**2)
    y = elastic_peak+loss_peak
    y = y/np.sum(y)
    doublet = {'x': incident_energy-np.flipud(energy_loss),
               'y': np.flipud(y),
               'x_min': -0.2+incident_energy,
               'x_max': separation+0.2+incident_energy}
    return doublet

def get_georgi_rixs(energy_loss, incident_energy=INCIDENT_ENERGY):
    """Return RIXS spectrum made up by Georgi
    """
    p1 = np.exp(-((energy_loss-0)/0.02)**2)
    p2 = 6*np.exp(-((energy_loss-0.1)/0.02)**2)
    p3 = 1.5*np.exp(-((energy_loss-0.2)/0.05)**2)
    p4 = 2*np.exp(-((energy_loss-0.5)/0.03)**2)
    p5 = 1.75*np.exp(-((energy_loss-0.6)/0.1)**2)
    p6 = 1.5*np.exp(-((energy_loss-0.9)/0.1)**2)
    y = p1+p2+p3+p4+p5+p6
    georgi_rixs = {'x': incident_energy-np.flipud(energy_loss),
                   'y': np.flipud(y),
                   'x_min': -0.5+incident_energy,
                   'x_max': 1.5+incident_energy}
    return georgi_rixs

def get_magnon_rixs(energy_loss, incident_energy=INCIDENT_ENERGY):
    """Return RIXS spectrum to model magnon
    Parameters chosen to approximate the spectrum shown in Fig. 4c of
    J. Kim et al., "Magnetic Excitation Spectra of Sr_2IrO_4 Probed by
    Resonant Inelastic X-Ray Scattering: Establishing Links to Cuprate
    Superconductors"
    PRL 108 177003 (2012)
    """
    p1 = 0.5*np.exp(-((energy_loss-0)/0.05)**2)
    p2 = 2*np.exp(-((energy_loss-0.2)/0.1)**2)
    p3 = 2*np.exp(-((energy_loss-0.55)/0.1)**2)
    p4 = 2*np.exp(-((energy_loss-0.75)/0.25)**2)
    y = p1+p2+p3+p4
    magnon_rixs = {'x': incident_energy-np.flipud(energy_loss),
                   'y': np.flipud(y),
                   'x_min': -0.5+incident_energy,
                   'x_max': 1.5+incident_energy}
    return magnon_rixs

def get_schlappa_rixs(energy_loss, incident_energy=INCIDENT_ENERGY):
    """Return RIXS spectrum to approximate that measured in Schlappa et al.
    Parameters were chosen to approximate the spectrum shown in Fig. 2b of
    J. Schlappa et al., "Spin-orbital separation in the quasi-one-dimensional
    Mott insulator Sr_2CuO_3" 
    Nature 485, 82-85 (2012)
    """
    p1 = 8*np.exp(-((energy_loss-0.25)/0.05)**2)
    p2 = 23*np.exp(-((energy_loss-1.8)/0.2)**2)
    p3 = 26*np.exp(-((energy_loss-2.2)/0.2)**2)
    p4 = 9*np.exp(-((energy_loss-2.9)/0.3)**2)
    p5 = 3*np.exp(-((energy_loss-4.5)/0.5)**2)
    p6 = 3*np.exp(-((energy_loss-5.2)/0.75)**2)
    y = p1+p2+p3+p4+p5+p6
    y = np.flipud(y)
    y = np.clip(y, 0, None)
    schlappa_rixs = {'x': incident_energy-np.flipud(energy_loss),
                     'y': y,
                     'x_min': -0.5+incident_energy,
                     'x_max': 7+incident_energy}
    return schlappa_rixs