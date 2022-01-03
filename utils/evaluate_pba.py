#!/usr/bin/env python
# coding: utf-8

# Evaluation: Transformer Dose Calculation 
## Import libraries and define auxiliary functions
import h5py
import numpy as np
import sys
sys.path.append('../src')
from preprocessing import DataRescaler
from evaluation import gamma_analysis, error_analysis

# Prepare input data.
path = '../data/training/'
path_test = '../data/test/'
path_weights = '../weights/weights.ckpt'
filename_test = path_test + 'test.h5'
filename_pba = path_test + 'testPBA.h5'
with h5py.File(filename_test, 'r') as fh:
    testIDs = [*range(fh['geometry'].shape[-1])]

# Load normalization constants.
scaler = DataRescaler(path)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':70, 'e_max':220}

# Gamma evaluation.
indexes_gamma, gamma_pass_rate, gamma_dist = gamma_analysis(
    model=filename_pba,
    testIDs=testIDs,
    filename=filename_test,
    scale=scale,
    num_sections=4,
    cutoff=0.1,
    inference=False
)
np.savez('./eval/gamma_analysis_pba.npz', indexes_gamma, gamma_pass_rate, gamma_dist)

# Error evaluation.
indexes_error, errors, error_dist = error_analysis(
    model=filename_pba,
    testIDs=testIDs,
    filename=filename_test,
    scale=scale,
    num_sections=4,
    cutoff=0.1,
    inference=False
)
np.savez('./eval/error_analysis_pba.npz', indexes_error, errors, error_dist)




