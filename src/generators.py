# -*- coding: utf-8 -*-
# Generator classes to dynamically load the data.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import glob
import h5py
import random
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.ndimage import shift

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
    def __init__(self, path, list_IDs, batch_size, scale, shuffle=True):

        self.path = path
        self.batch_size = batch_size
        self.list_IDs = list_IDs 
        self.shuffle = shuffle
        self.on_epoch_end()

        # Get input and output dimensions
        # TODO: won't be needed with beam shape, this step basically
        # reduces the original data dimension (e.g., (25, 25)) to something
        # that is convolutionally friendly (e.g., (24, 24))
        self.input_dim = tuple(np.load(self.path+'0.npz')['vol'].shape)
        self.output_dim = tuple(np.load(self.path+'0.npz')['dose'].shape)

        # If Height = Width rotate 90 degrees, else 180
        self.rotk = np.arange(4) if self.input_dim[-1]==self.input_dim[-2] else [0,2]
        self.shift = - self.input_dim[0] // 2

        # Load scaling factors
        self.x_min = scale['x_min']
        self.x_max = scale['x_max']
        self.y_min = scale['y_min']
        self.y_max = scale['y_max']
        self.r_min = scale['r_min']
        self.r_max = scale['r_max']
        
    def __len__(self):
        """ Calculates the number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples
        X = np.empty((self.batch_size, *self.input_dim))
        r = np.empty((self.batch_size, *self.input_dim))
        y = np.empty((self.batch_size, *self.output_dim))

        # Load batch of data
        for i, ID in enumerate(list_IDs_temp):

            # Random rotation of input and output
            rot = np.random.choice(self.rotk)
            sft = random.randint(self.shift, 0)
            X[i,] = shift(
                np.rot90(np.load(self.path+str(ID)+'.npz')['vol'], rot, (1,2)),[sft, 0, 0], order=0)
            r[i,] = shift(
                np.rot90(np.load(self.path+str(ID)+'.npz')['ray'], rot, (1,2)), [sft, 0, 0], order=0)
            y[i,] = shift(
                np.rot90(np.load(self.path+str(ID)+'.npz')['dose'], rot, (1,2)), [sft, 0, 0], order=0)
                
        # Normalize dose, ray tracing and intensity values separately
        X = (X - self.x_min) / (self.x_max - self.x_min)
        r = (r - self.r_min) / (self.r_max - self.r_min)
        y = (y - self.y_min) / (self.y_max - self.y_min)

        return [np.expand_dims(X, -1), np.expand_dims(r, -1)], np.expand_dims(y, -1)
