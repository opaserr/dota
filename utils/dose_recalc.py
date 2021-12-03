# coding: utf-8
## Import libraries and define auxiliary functions
import h5py
import json
import numpy as np
from ..src.models import multi_energy_model
from ..src.preprocessing import DataRescaler
from ..src.evaluation import gamma_analysis

# Load model and data hyperparameters
num_beams = 2
batch_size = 128
with open('../hyperparam.json', 'r') as hfile:
    param = json.load(hfile)
    
# Calculate or load normalization constants.
scaler = DataRescaler(path, filename=filename)
scaler.load(inputs=True, outputs=True)
scale = {'y_min':scaler.y_min, 'y_max':scaler.y_max,
        'x_min':scaler.x_min, 'x_max':scaler.x_max,
        'e_min':70, 'e_max':220}

# Define and load the model.
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
transformer.load_weights(path_ckpt)

# Iterate over beams
path_inputs = '..data/plans/beam_'
path_outputs = '..data/plans/dose_'

for b in range(num_beams):

    # Load beam data and patient geometries
    with h5py.File(path_inputs + str(b) + '.h5', 'r') as fh:
        weights = fh['weights'][:] / 10e3
        num_energies = fh['num_energies'][:]
        energies = fh['energy'][:]
        energies = (energies - scale['e_min']) / (scale['e_max'] - scale['e_min'])
        geometries = np.transpose(fh['geometry'][:])
        geometries = (geometries - scale['x_min']) / (scale['x_max'] - scale['x_min'])

    # Expand input arrays with as many input geometries as energies
    ind = np.append(0, np.cumsum(num_energies))
    inputs = np.empty((np.sum(num_energies), *tuple(geometries.shape[1:])))
    for i, num in enumerate(num_energies):
        inputs[ind[i]:ind[i+1],] = np.repeat(geometries[i], num, axis=0)

    # Predict dose distribution
    outputs = np.squeeze(model.predict(
        [np.expand_dims(inputs, -1), np.expand_dims(energies, -1)]))
    outputs = outputs * (scale['y_max'] - scale['y_min']) + scale['y_min']

    # Multiply by weights
    outputs = np.transpose(np.transpose(outputs) * weights)

    # Accumulate dose per geometry
    doses = np.empty_like(geometries)
    for i, num in enumerate(num_energies):
        doses[i,] = np.sum(outputs[ind[i]:ind[i+1],], axis=0)

    # Save beam doses to .h5 file
    with h5py.File(path_outputs + str(b) + '.h5', 'r') as fh:
        fh.create_dataset('dose', data=np.transpose(doses))
