#!/usr/bin/env python
# coding: utf-8
# Function to create a hyperparameter dictionary
# that is read by other functions and classes.
import json

# Data and model parameters
param = dict()
param['num_slices'] = 150
param['data_shape'] = (24, 24, 1)
param['num_tokens'] = param['num_slices'] + 1
param['kernel_size'] = 5
param['num_heads'] = 16
param['num_transformers'] = 1
param['projection_dim'] = (6, 6, 16)

with open('../hyperparam.json', 'w') as write_file:
    json.dump(param, write_file)
