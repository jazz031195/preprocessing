import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nibabel as nib
import os
from os import listdir
from os.path import isfile, join
from scipy import signal
from pydfMRI.imaging_tools import remove_dummy_volumes, signal_detrend, image_padding, clean_data, topup, motion_correction, calculate_adc
from pathlib import Path
import json

with open('args.json') as f:
    args = json.load(f)

def print_step(to_print):
    """
    Function to print the different steps of the pipeline 
    
    """

    print('\n')
    print('*'*100)
    print(to_print)
    print('*'*100)
    print('\n')


data_directory = args['data_directory']
subject_names  = args['subject_names']
# Which step(s) of the pipeline to run
steps_to_run   = args['steps_to_run']
# Number of volumes to remove from the timeserie beginning
nb_dummy_vols  = args['nb_dummies']

for i, subject in enumerate(subject_names):
    subject_path = Path(data_directory) / subject
    dwi_raw_path = subject_path / 'dwi' / f'{subject}-dwi-200-1000.nii.gz'

    # Removes the first dummy volumes & the outliers from the dwi_raw
    dwi_clean_path = subject_path / 'preprocessed/wo_outliers' / f'{subject}-dwi-200-1000_clean.nii.gz'
    if (not dwi_clean_path.exists()) | ("cleaning" in steps_to_run) | ("all" in steps_to_run):
        print_step(f"Cleaning on...{dwi_raw_path}\nOutput in.....{dwi_clean_path}")
        clean_data(dwi_raw_path, dwi_clean_path, nb_dummy_vols[i])

    # MPPCA denoising
    dwi_denoised_path = subject_path / 'preprocessed/denoised' / f'{subject}-dwi-200-1000_denoised.nii.gz'
    if (not dwi_denoised_path.exists()) | ("mppca" in steps_to_run) | ("all" in steps_to_run):
        print_step(f"MPPCA on.....{dwi_clean_path}\nOutput in....{dwi_denoised_path}")
        dwi_denoised_sigma = subject_path / 'preprocessed/denoised' / f'{subject}-dwi-200-1000_denoised_sigma.nii.gz'
        os.system(f'dwidenoise {dwi_clean_path} {dwi_denoised_path} -noise {dwi_denoised_sigma} -extent 9 -force')

    # Gibbs unringing
    dwi_denoised_unringed_path = subject_path / 'preprocessed/unringed' / f'{subject}-dwi-200-1000_denoised_unringed.nii.gz'
    if (not dwi_denoised_unringed_path.exists()) | ("gibbs" in steps_to_run) | ("all" in steps_to_run):
        print_step(f"Gibbs unringing on..{dwi_denoised_path}\nOutput in...........{dwi_denoised_unringed_path}")
        os.system(f'mrdegibbs {dwi_denoised_path} {dwi_denoised_unringed_path}')

    # Topup
    dwi_denoised_unringed_tu_path = subject_path / 'preprocessed/topup' / f'{subject}-dwi-200-1000_denoised_unringed_sdc.nii.gz'
    if (not dwi_denoised_unringed_tu_path.exists()) | ("topup" in steps_to_run) | ("all" in steps_to_run):
        print_step(f"Topup on.....{dwi_denoised_unringed_path}\nOutput in....{dwi_denoised_unringed_tu_path}")
        dwi_RPE = subject_path / f'dwi/{subject}-dwi-200-1000-revPE.nii.gz'
        topup(dwi_denoised_unringed_path, dwi_RPE, subject_path)


    # Motion correction
    dwi_denoised_unringed_tu_MC_path = subject_path / 'preprocessed/ANTS' / f'{subject}-dwi-200-1000_denoised_unringed_sdc_mc'
    if (not dwi_denoised_unringed_tu_MC_path.with_suffix('.nii.gz').exists()) | ("MC" in steps_to_run) | ("all" in steps_to_run):
        print_step(f"Motion correction on..{dwi_denoised_unringed_tu_path}\nOutput in.............{dwi_denoised_unringed_tu_MC_path}")
        motion_correction(dwi_denoised_unringed_tu_path, dwi_denoised_unringed_tu_MC_path)

    # ADC
    dwi_denoised_unringed_tu_MC_path = dwi_denoised_unringed_tu_MC_path.with_suffix('.nii.gz')
    adc_path                         = subject_path / 'preprocessed/adc' / f'{subject}-adc.nii.gz'
    if (not adc_path.exists()) | ("adc" in steps_to_run) | ("all" in steps_to_run):
        print_step(f"ADC on..........{dwi_denoised_unringed_tu_MC_path} \nOutput in.......{adc_path}")
        calculate_adc(dwi_denoised_unringed_tu_MC_path, adc_path)