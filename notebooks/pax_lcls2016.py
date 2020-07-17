#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:23:08 2018

@author: dhigley
"""

import numpy as np

PIXELS_TO_EV = 1.0/38.61111111

runs_to_import = [17, 21, 22, 23, 24, 25, 27, 28, 29, 30, 32]
fname_start = 'lcls_data/run00'
blob_data = {}
for run in runs_to_import:
    fname_end = str(run)+'_thres0006_Blob'
    fname = fname_start+fname_end
    blob_data[str(run)] = np.genfromtxt(fname)
threshold_data = {}
for run in runs_to_import:
    fname_end = str(run)+'_thres0045_Threshold'
    fname = fname_start+fname_end
    threshold_data[str(run)] = np.genfromtxt(fname)
    #threshold_data[str(run)] = threshold_data[str(run)]-np.mean(threshold_data[str(run)][800:])

INCIDENT_PHOTON_ENERGY = {  '17': 782.7,
                            '24': 774,
                            '23': 775,
                            '22': 776,
                            '21': 777,
                            '25': 778,
                            '27': 779,
                            '28': 780,
                            '32': 780,
                            '29': 781,
                            '30': 782}
ANALYZER_KINETIC_ENERGY = {'17': 692,
                           '21': 688,
                           '22': 688,
                           '23': 688,
                           '24': 688,
                           '25': 688,
                           '27': 693,
                           '28': 693,
                           '29': 693,
                           '30': 693,
                           '32': 693}
VERNIER = {'21': -1,
           '22': -3,
           '23': -5,
           '24': -7,
           '25': 3,
           '27': 5,
           '28': 8,
           '29': 11,
           '30': 14}

ELASTIC_PEAK = {}
for run in INCIDENT_PHOTON_ENERGY.keys():
    ELASTIC_PEAK[run] = 7.8+INCIDENT_PHOTON_ENERGY[run]-774+ANALYZER_KINETIC_ENERGY[run]-698.7
    
def get_recorded_au4f_spectrum():
    psf_intensity = threshold_data['17'][100:]
    psf_kinetic_energy = (np.arange(len(psf_intensity))+100)*PIXELS_TO_EV+ANALYZER_KINETIC_ENERGY['17']
    psf_binding_energy = INCIDENT_PHOTON_ENERGY['17']-psf_kinetic_energy
    psf = {'x': psf_kinetic_energy,
           'y': psf_intensity,
           'binding_energy': psf_binding_energy}
    return psf
    
def get_extended_au4f_spectrum(energy_spacing, num_points):
    psf = get_recorded_au4f_spectrum()
    # center of Au 4f doublet (determined by looking at plotted spectrum):
    center_au4f = 709
    energies = np.arange(num_points)*energy_spacing
    energies = energies-np.mean(energies)+center_au4f
    psf_binding_energy = INCIDENT_PHOTON_ENERGY['17']-energies
    left = np.mean(psf['y'][psf['x'] < 700])
    right = np.mean(psf['y'][psf['x'] > 715])
    extended_au4f = {'x': energies,
                     'y': np.interp(energies, psf['x'], psf['y'], left=left,
                                    right=right),
                     'binding_energy': psf_binding_energy}
    return extended_au4f
    
    
def get_lcls_specs():
    """Return recorded LCLS PAX spectra
    """
    # runs in ascending order of incident photon energy:
    runs_to_get = ['24', '23', '22', '21', '25', '27', '28', '32', '29', '30']
    spectra = []
    incident_photon_energies = []
    for run in runs_to_get:
        intensity = threshold_data[run][100:]
        spec = {'x': (np.arange(len(intensity))+100)*PIXELS_TO_EV+ANALYZER_KINETIC_ENERGY[run],
                'y': intensity}
        spectra.append(spec)
        incident_photon_energies.append(INCIDENT_PHOTON_ENERGY[run])
    energy_spacing = np.mean(np.diff(spectra[0]['x']))
    num_points = len(spectra[0]['x'])*2-1
    psf = get_extended_au4f_spectrum(energy_spacing, num_points)
    #psf = get_recorded_au4f_spectrum()
    to_return = {'psf': psf,
                 'spectra': spectra,
                 'incident_photon_energy': incident_photon_energies}
    return to_return

# energy_loss = np.arange(len(deconvolved_spec['deconvolved_spec']))*PIXELS_TO_EV-ELASTIC_PEAK[run] 
