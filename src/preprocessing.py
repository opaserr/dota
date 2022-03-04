# -*- coding: utf-8 -*-
# Auxiliary preprocessing methods.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import glob
import h5py
import numpy as np

class DataRescaler():
    """
    Creates and object that contains the rescaling factors
    to get the data in the interval [0,1].

    *path: string path to the data. Recommended: data/training/.
    """
    def __init__(self, path, filename=None, ikey='geometry', okey='dose'):
        self.path = path
        self.ikey = ikey
        self.okey = okey
        self.filename = path + 'train.h5' if filename is None else filename 
        # Initialize to [0,1]
        self.x_min, self.x_max = 0, 1
        self.y_min, self.y_max = 0, 1

    def __get_values(self, key):
        '''Calculate the maximum and minimum.'''
        # Initialize max and min values.
        max_value, min_value = -np.float32('inf'), np.float32('inf')

        with h5py.File(self.filename, 'r') as fh:
            for i in range(fh[key].shape[-1]):
                X = fh[key][:,:,:,i]
                max_value = max(max_value, np.max(X))
                min_value = min(min_value, np.min(X))
                # TODO: add energy dependence

        # Save values to .txt file
        with open(self.path + 'minmax_' + key + '.txt', 'w') as f:
            f.write(f'{min_value, max_value}')

        return min_value, max_value

    def __load_values(self, key):
        # Read values from .txt file
        with open(self.path + 'minmax_' + key + '.txt', 'r') as f:
            min_value, max_value = eval(f.readline())

        return min_value, max_value

    def fit(self, inputs=False, outputs=False):
        # Calculate and update min/max values
        if inputs:
            self.x_min, self.x_max = self.__get_values(self.ikey)
        if outputs:
            self.y_min, self.y_max = self.__get_values(self.okey)

    def load(self, inputs=False, outputs=False):
        # Load previously calculated values
        if inputs:
            self.x_min, self.x_max = self.__load_values(self.ikey)
        if outputs:
            self.y_min, self.y_max = self.__load_values(self.okey)
