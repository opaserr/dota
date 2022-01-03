# coding: utf-8
## Import libraries and define auxiliary functions
import h5py
import json
import numpy as np
import time
import sys
sys.path.append('../src')
from models import dota_energies
from preprocessing import DataRescaler
from evaluation import gamma_analysis

start = time.perf_counter()

# Load model and data hyperparameters
patient = 1
num_beams = 2
batch_size = 8
with open('../hyperparam.json', 'r') as hfile:
    param = json.load(hfile)
    
# Calculate or load normalization constants.
path = '../data/training/'
path_ckpt = '../weights/weights.ckpt'
scaler = DataRescaler(path)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':70, 'e_max':220}

# Define and load the model.
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
transformer.load_weights(path_ckpt)

# Iterate over beams
path_inputs = f'../data/plans/P{patient}/beam_'
path_outputs = f'../data/plans/P{patient}/dose_'

for b in range(num_beams):

    # Load beam data and patient geometries
    with h5py.File(path_inputs + str(b+1) + '.h5', 'r') as fh:
        weights = fh['weights'][:] / 10e3
        num_energies = np.squeeze(fh['num_energies'][:]).astype(int)
        energies = np.squeeze(fh['energy'][:])
        energies = (energies - scale['e_min']) / (scale['e_max'] - scale['e_min'])
        geometries = np.transpose(fh['geometry'][:])
        geometries = (geometries - scale['x_min']) / (scale['x_max'] - scale['x_min'])

    # Expand input arrays with as many input geometries as energies
    ind = np.append(0, np.cumsum(num_energies))
    inputs = np.empty((int(np.sum(num_energies)), *tuple(geometries.shape[1:])))
    for i, num in enumerate(num_energies):
        inputs[ind[i]:ind[i+1],:,:,:] = np.repeat(np.expand_dims(geometries[i], 0), num, axis=0)

    # Predict dose distribution
    outputs = np.squeeze(transformer.predict(
        [np.expand_dims(inputs, -1), np.expand_dims(energies, -1)]))
    outputs = outputs * (scale['y_max'] - scale['y_min']) + scale['y_min']
    outputs[outputs<0.1] = 0 # TODO: make cutoff variable

    # Multiply by weights
    outputs = np.transpose(np.transpose(outputs) * weights)

    # Accumulate dose per geometry
    doses = np.empty_like(geometries)
    for i, num in enumerate(num_energies):
        doses[i,] = np.sum(outputs[ind[i]:ind[i+1],], axis=0)

    # Save beam doses to .h5 file
    with h5py.File(path_outputs + str(b+1) + '_dota.h5', 'w') as fh:
        fh.create_dataset('dose', data=np.transpose(doses))

print(time.perf_counter() - start)