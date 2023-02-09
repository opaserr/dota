#!/usr/bin/env python
# coding: utf-8

# Transformer Dose Calculation 
## Import libraries and define auxiliary functions
import h5py
import json
import random
import sys
sys.path.append('./src')
import numpy as np
from generators import DataGenerator
from models import dota_residual, dota_photons
from tensorflow_addons.optimizers import LAMB
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.config import list_physical_devices
print(list_physical_devices('GPU'))

## Define hyperparameters
# Training parameters
batch_size = 4
num_epochs = 120
learning_rate = 0.001
weight_decay = 0.0001

# Load model and data hyperparameters
with open('./hyperparam.json', 'r') as hfile:
    param = json.load(hfile)
    
# Load data files
path = './data/'
path_ckpt = './weights/ckpt/weights.ckpt'
train_split = 0.90
listIDs = [*range(1700)]

# Training, validation, test split.
random.seed(333)
random.shuffle(listIDs)
trainIDs = listIDs[:int(round(train_split*len(listIDs)))]
valIDs = listIDs[int(round(train_split*len(listIDs))):]
    
# Calculate or load normalization constants.
scale = {'y_min':0, 'y_max':3.0755550861358643,
        'r_min':0, 'r_max':3.0755550861358643,
        'x_min':0, 'x_max':4.071000099182129}

# Initialize generators.
train_gen = DataGenerator(path, trainIDs, batch_size, scale)
val_gen = DataGenerator(path, valIDs, batch_size, scale)

## Define and train the transformer.
transformer = dota_residual(
    inshape=param['inshape'],
    steps=param['num_levels'],
    enc_feats=param['enc_feats'],
    num_heads=param['num_heads'],
    num_transformers=param['num_transformers'],
    kernel_size=param['kernel_size']
)
transformer.summary()

# Load weights from checkpoint.
random.seed()
#transformer.load_weights(path_ckpt)

# Compile the model.
optimizer = LAMB(learning_rate=learning_rate, weight_decay_rate=weight_decay)
transformer.compile(optimizer=optimizer, loss='mse', metrics=[])

# Callbacks.
# Save best model at the end of the epoch.
checkpoint = ModelCheckpoint(
    filepath=path_ckpt,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min')

# Learning rate scheduler. Manually reduce the learning rate.
sel_epochs = [10,25,40,55,70,85,100,115]
lr_scheduler = LearningRateScheduler(
    lambda epoch, lr: lr*0.5 if epoch in sel_epochs else lr,
    verbose=1)

optimizer.learning_rate.assign(learning_rate)
history = transformer.fit(
    x=train_gen,
    validation_data=val_gen,
    epochs=num_epochs,
    verbose=2,
    callbacks=[checkpoint, lr_scheduler]
    )

# Save last weights and hyperparameters.
path_last = './weights/weights.ckpt'
transformer.save_weights(path_last)
