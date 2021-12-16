# -*- coding: utf-8 -*-
# Generator classes to dynamically load the data.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import glob
import h5py
import random
import numpy as np
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Generates data for Keras.
    Assumes that the data is stored in a folder (e.g. /data) 
    inside a single .h5 file.

    *list_IDs: a list with the file identifiers.
    *batch_size: size of the mini-batch.
    *num_energies: number of different energies per geometry.
    *ikey, okey: input and output key identifiers.
    *shuffle: flag to shuffle IDs after each epoch.
    *scale: dict with max and min values
    """
    def __init__(self, list_IDs, batch_size, path, scale, num_energies=2, 
                 ikey='geometry', okey='dose', shuffle=True):

        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.path = path 
        self.ikey = ikey
        self.okey = okey
        self.shuffle = shuffle
        self.num_energies = num_energies
        self.on_epoch_end()

        # Get input and output dimensions
        # TODO: won't be needed with beam shape, this step basically
        # reduces the original data dimension (e.g., (25, 25)) to something
        # that is convolutionally friendly (e.g., (24, 24))
        with h5py.File(self.path, 'r') as fh:
            input_shape = tuple(reversed(fh[self.ikey].shape[:-1]))
            self.input_dim = tuple(map(lambda i, j: i - j, input_shape, (0,1,1)))
        with h5py.File(self.path, 'r') as fh:
            output_shape = tuple(reversed(fh[self.okey+'0'].shape[:-1]))
            self.output_dim = tuple(map(lambda i, j: i - j, output_shape, (0,1,1)))

        # If Height = Width rotate 90 degrees, else 180
        self.rotk = np.arange(4) if self.input_dim[-1]==self.input_dim[-2] else [0,2]

        # Load scaling factors
        self.x_min = scale['x_min']
        self.x_max = scale['x_max']
        self.y_min = scale['y_min']
        self.y_max = scale['y_max']
        self.min_energy = scale['e_min']
        self.max_energy = scale['e_max']
        
    def __len__(self):
        # Calculates the number of batches per epoch.
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data.
        # Generate indexes of the batch.
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs.
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data.
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch.
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples.
        X = np.empty((self.batch_size, *self.input_dim))
        y = np.empty((self.batch_size, *self.output_dim))
        energies = np.empty((self.batch_size))

        # Load data.
        with h5py.File(self.path, 'r') as fh:
            for i, ID in enumerate(list_IDs_temp):
                dose_index = random.choice(list(range(self.num_energies)))
                tmpGeometry = np.transpose(fh[self.ikey][:,:,:,ID])
                tmpDose = np.transpose(fh[self.okey+str(dose_index)][:,:,:,ID])
                energies[i] = fh['energy'+str(dose_index)][ID]

                # Augment data.
                # TODO: remove last slicing for shapes (see above)
                rot = np.random.choice(self.rotk)
                X[i,] = np.rot90(tmpGeometry, rot, (1,2))[:,:-1,:-1] 
                y[i,] = np.rot90(tmpDose, rot, (1,2))[:,:-1,:-1]
                
        X = (X - self.x_min) / (self.x_max - self.x_min)
        y = (y - self.y_min) / (self.y_max - self.y_min)
        energies = (energies - self.min_energy) / (self.max_energy - self.min_energy)

        return [np.expand_dims(X, -1), np.expand_dims(energies, -1)], np.expand_dims(y, -1)
