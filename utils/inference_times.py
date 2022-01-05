#!/usr/bin/env python
# coding: utf-8

# Evaluation: Transformer Dose Calculation 
## Import libraries and define auxiliary functions
import h5py
import json
import numpy as np
import sys
sys.path.append('../src')
from models import dota_energies
from preprocessing import DataRescaler
from evaluation import time_analysis

batch_size = 8

# Load model and data hyperparameters.
with open('../hyperparam.json', 'r') as hfile:
    param = json.load(hfile)

# Prepare input data.
path = '../data/training/'
path_test = '../data/test/'
path_weights = '../weights/weights.ckpt'
filename_test = path_test + 'test.h5'
with h5py.File(filename_test, 'r') as fh:
    testIDs = [*range(fh['geometry'].shape[-1])]

# Load normalization constants.
scaler = DataRescaler(path)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':70, 'e_max':220}

## Define and load the transformer model.
transformer = dota_energies(
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

times = time_analysis(
    model=transformer,
    testIDs=testIDs,
    filename=filename_test,
    scale=scale,
    batch_size=batch_size
)
np.savez('./eval/time_analysis.npz', times)
