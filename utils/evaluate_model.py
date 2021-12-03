#!/usr/bin/env python
# coding: utf-8

# Evaluation: Transformer Dose Calculation 
## Import libraries and define auxiliary functions
import h5py
import json
import numpy as np
import sys
sys.path.append('../src')
from models import multi_energy_model
from preprocessing import DataRescaler
from evaluation import gamma_analysis, error_analysis

# Load model and data hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)

# Prepare input data.
path = '../data/training/'
path_test = '../data/test/'
path_weights = '../weights/weights.ckpt'
filename_test = path_test + 'test.h5'
with h5py.File(filename_test, 'r') as fh:
    testIDs = [*range(fh['geometry'].shape[-1])]

# Load normalization constants
scaler = DataRescaler(path)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':70, 'e_max':220}

## Define and load the transformer.
transformer = multi_energy_model(
    num_tokens=param['num_tokens'],
    input_shape=param['data_shape'],
    projection_dim=param['projection_dim'],
    num_heads=param['num_heads'],
    num_transformers=param['num_transformers'], 
    kernel_size=param['kernel_size'],
    causal=True
)
transformer.summary()

# Load weights from checkpoint.
transformer.load_weights(path_weights)

# Gamma evaluation.
indexes_gamma, gamma_pass_rate, gamma_dist = gamma_analysis(
    model=transformer,
    testIDs=testIDs,
    filename=filename_test,
    scale=scale,
    num_sections=4,
    cutoff=0,
    energy=True
)
np.savez('./eval/gamma_analysis.npz', indexes_gamma, gamma_pass_rate, gamma_dist)

# Error evaluation.
indexes_error, errors, error_dist = error_analysis(
    model=transformer,
    testIDs=testIDs,
    filename=filename_test,
    scale=scale,
    num_sections=4,
    cutoff=0,
    energy=True
)
np.savez('./eval/error_analysis.npz', indexes_error, errors, error_dist)
