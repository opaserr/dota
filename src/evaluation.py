# -*- coding: utf-8 -*-
# Auxiliary functions to evaluate the models.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import h5py
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from pymedphys import gamma

def infer(model, ID, path, scale, ikey='vol', okey='dose', rkey='ray', cutoff=0.5):
    """
    Get model prediction from test sample ID.
    """
    # Load test sample input and ground truth
    geometry = np.expand_dims(np.load(path+str(ID)+'.npz')[ikey], axis=(0,-1))
    ct_vol = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])
    raytrace = np.expand_dims(np.load(path+str(ID)+'.npz')[rkey], axis=(0,-1))
    ray_tr = (raytrace - scale['r_min']) / (scale['r_max'] - scale['r_min'])
    ground_truth = np.load(path+str(ID)+'.npz')[okey]

    # Predict dose distribution
    prediction = model.predict([ct_vol, ray_tr])
    prediction = prediction * (scale['y_max']-scale['y_min']) + scale['y_min']
    prediction[prediction<(cutoff/100)*scale['y_max']] = 0

    return np.squeeze(geometry), np.squeeze(prediction), np.squeeze(ground_truth)

def from_file(path, ID, gt_path, ikey='vol', okey='dose'):
    """
    Load sample with ID from an alternative filename instead of
    using the model. Uses for comparison.
    """
    # Load test sample input and ground truth
    inputs = np.load(gt_path+str(ID)+'.npz')[ikey], axis=(0,-1)
    ground_truth = np.load(gt_path+str(ID)+'.npz')[okey], axis=(0,-1)

    # Load sample from alternative filename
    prediction = np.load(path+str(ID)+'.npz')[okey], axis=(0,-1)

    return np.squeeze(inputs), np.squeeze(prediction), np.squeeze(ground_truth)

def inference_time(model, IDs, path, scale, batch_size, input_dim,
    ikey='vol', rkey='ray'):
    """
    Get time elapsed to predict a dose distribution.
    """
    ct_vol = np.empty((batch_size, *input_dim, 1))
    ray_tr = np.empty((batch_size, *input_dim, 1))

    for i, ID in enumerate(IDs):
        # Load test sample input and ground truth
        ct_vol[i,] = np.expand_dims(np.load(path+str(ID)+'.npz')[ikey], axis=(0,-1))
        ct_vol[i,] = (ct_vol[i,] - scale['x_min']) / (scale['x_max'] - scale['x_min'])
        ray_tr[i,] = np.expand_dims(np.load(path+str(ID)+'.npz')[rkey], axis=(0,-1))
        ray_tr[i,] = (ray_tr[i,] - scale['r_min']) / (scale['r_max'] - scale['r_min'])

    # Predict dose distribution
    start = time.perf_counter()
    prediction = model.predict([ct_vol, ray_tr], batch_size=batch_size)
    end = time.perf_counter()

    elapsed_time = end - start

    return elapsed_time

def estimate_range(ground_truth, percentile=5):
    """
    Estimate the range of the beam based on the last non zero layer.
    Optionally zero out dose values below a percentile.
    """
    # Zero out values lower than the specified percentile
    ground_truth[ground_truth<np.percentile(ground_truth, percentile)] = 0

    # Reshape array to 2D
    target_shape = (ground_truth.shape[0], np.prod(ground_truth.shape[1:]))
    ground_truth = np.reshape(ground_truth, target_shape)

    # Return last index with values above 0.
    return max(np.where(ground_truth.any(axis=1))[0])

def gamma_analysis(model, testIDs, path, scale, num_sections=1,
    dose_threshold=1, distance_threshold=3, cutoff=0, resolution=[2,2,2],
    ikey='vol', okey='dose', inference=True):
    """
    Performs a gamma analysis. Optionally calculates in which
    part of the beam (quadrant) the failed voxels are.
    """
    # Initialize auxiliary variables.
    gamma_pass_rate = np.empty((2, len(testIDs)))
    gamma_dist = np.empty((len(testIDs), num_sections))
    num_IDs = len(testIDs)

    # Initial call to print 0% progress
    progress_bar(0, num_IDs, prefix="Progress:", suffix="Complete")

    # Loop over test samples.
    for i, ID in enumerate(testIDs):

        # Get input, output, ground_truth triplet.
        if inference:
            inputs, prediction, ground_truth = infer(model, ID, path, scale,
                ikey=ikey, okey=okey) 
        else:
            inputs, prediction, ground_truth = from_file(model, ID, path,
                ikey=ikey, okey=okey)

        # Cut off MC noise
        ground_truth[ground_truth<(cutoff/100)*scale['y_max']] = 0
        prediction[prediction<(cutoff/100)*scale['y_max']] = 0

        # Calculate gamma values.
        axes = (np.arange(ground_truth.shape[0])*resolution[0],
            np.arange(ground_truth.shape[1])*resolution[1],
            np.arange(ground_truth.shape[2])*resolution[2])
        gamma_values = gamma(axes, ground_truth, axes, prediction, dose_threshold,
            distance_threshold, lower_percent_dose_cutoff=0, quiet=True)
        gamma_values = np.nan_to_num(gamma_values, 0)

        # Conservative estimate using only dose voxels
        # gamma_values_c = gamma_values[~np.isnan(gamma_values)]
        gamma_pass_rate[0,i] = 1-(np.count_nonzero(
            gamma_values>1)/np.count_nonzero(gamma_values>0))

        # General estimate using all voxels
        gamma_pass_rate[1,i] = np.sum(gamma_values<=1)/np.prod(gamma_values.shape)

        # Calculate average error per quadrant
        if num_sections > 1:

            # Get beam depth.
            last_index = estimate_range(ground_truth)
            interval = math.floor(last_index / num_sections)
            ticks = list(range(0, last_index+1, interval))

            if len(ticks) is not num_sections+1:
                raise ValueError("Wrong quadrant limit calculation.")

            # Store the pass rate per section
            for j in range(1, len(ticks)):
                gamma_section = gamma_values[ticks[j-1]:ticks[j],:,:]
                gamma_dist[i,j-1] = np.sum(gamma_section>1)

        progress_bar(i+1, num_IDs, prefix="Progress:", suffix="Complete")

    # Sort list and indexes
    indexes = np.argsort(gamma_pass_rate)

    return indexes, gamma_pass_rate, gamma_dist

def error_analysis(model, testIDs, path, scale, num_sections=1,
    ikey='vol', okey='dose', cutoff=0, inference=True):
    """
    Calculates the average error in the test set, and the sample with
    the highest error. Optionally calculates in which part of the beam 
    (quadrant) the errors are.
    """
    # Initialize auxiliary variables.
    errors = np.empty((2, len(testIDs)))
    error_dist = np.empty((len(testIDs), num_sections))
    rmse = np.empty((len(testIDs), 1))
    num_IDs = len(testIDs)

    # Initial call to print 0% progress
    progress_bar(0, num_IDs, prefix="Progress:", suffix="Complete")

    # Loop over test samples.
    for i, ID in enumerate(testIDs):

        # Get input, output, ground_truth triplet.
        if inference:
            inputs, prediction, ground_truth = infer(model, ID, path, 
                ikey=ikey, okey=okey) 
        else:
            inputs, prediction, ground_truth = from_file(model, ID, path,
                ikey=ikey, okey=okey)         

        # Cut off MC noise 
        ground_truth[ground_truth<(cutoff/100)*scale['y_max']] = 0
        prediction[prediction<(cutoff/100)*scale['y_max']] = 0

        # Store test sample mean absolute error.
        absolute_error = np.absolute(prediction-ground_truth)
        errors[0,i] = absolute_error.mean()

        # Store relative error
        errors[1,i] = 100 * errors[0,i] / np.amax(ground_truth) 

        # Store root mean square error
        rmse[i] = np.sqrt(((prediction - ground_truth) ** 2).mean())

        # Calculate average error per quadrant
        if num_sections > 1:

            # Get beam depth.
            last_index = estimate_range(ground_truth)
            interval = math.floor(last_index / num_sections)
            ticks = list(range(0, last_index+1, interval))

            if len(ticks) is not num_sections+1:
                raise ValueError("Wrong quadrant limit calculation.")

            # Store the mae per section
            for j in range(1, len(ticks)):
                error_dist[i,j-1] = (absolute_error[ticks[j-1]:ticks[j],:,:]).mean()

            error_dist[i,] = 100 * error_dist[i,]/ np.amax(ground_truth)

        progress_bar(i+1, num_IDs, prefix="Progress:", suffix="Complete")

    # Sort list and indexes
    indexes = np.argsort(errors)

    return indexes, errors, error_dist, rmse

def time_analysis(model, testIDs, path, scale, ikey='vol', okey='dose',
    batch_size=1):
    """
    Calculates the inference time across the test dataset.
    """
    # Initialize auxiliary variables.
    num_batches = math.floor(len(testIDs)/batch_size)
    times = np.empty((num_batches,))

    input_dim = tuple(np.load(self.path+'0.npz')[ikey].shape)

    # Initial call to print 0% progress
    progress_bar(0, num_batches, prefix="Progress:", suffix="Complete")

    # Loop over test samples.
    for i in range(num_batches):

        IDs = testIDs[i*batch_size:(i+1)*batch_size]

        # Get inference time.
        times[i,] = inference_time(model, IDs, path, scale, batch_size,
            input_dim, ikey) 

        progress_bar(i+1, num_batches, prefix="Progress:", suffix="Complete")

    return times / batch_size

def progress_bar(iteration, total, prefix='', suffix='', decimals=1,
    length=50, fill='=', printEnd='\r'):
    """
    Call within a loop to create terminal progress bar
    iteration.......current iteration
    total...........total number of iterations
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
