# -*- coding: utf-8 -*-
# Auxiliary functions to plot data.
# COPYRIGHT: TU Delft, Netherlands. 2021.
# LICENSE: GNU AGPL-3.0 
import h5py
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from pymedphys import gamma

def plot_beam(inputs, ground_truth, outputs, gamma_evaluation=False, slices=10, 
    figsize=(20,30), gamma_cutoff=0.1, fontsize=10, resolution=[2.5,2.5,2.5]):
    """
    Plots slices of the full beam along the Z axis.
    *inputs..........3D array [Y,X,Z] from function infer
    """
    # Initialize figure and axes.
    num_cols = 4
    first_slice = np.floor((outputs.shape[-1] - slices) / 2)
    fig, axs = plt.subplots(slices, num_cols, figsize=figsize,
                            sharey = True, sharex=True)
    axs[0,0].set_title("CT scan", fontsize=fontsize)
    axs[0,1].set_title("Target", fontsize=fontsize)
    axs[0,2].set_title("Predicted", fontsize=fontsize)

    if gamma_evaluation:
        axs[0,3].set_title("Gamma analysis", fontsize=fontsize)
    else:
        axs[0,3].set_title("Difference", fontsize=fontsize)

    # Calculate maximum and minimum per column.
    min_input, max_input = np.min(inputs), np.max(inputs)
    min_output, max_output = np.min(outputs), np.max(outputs)
    selected_slices = np.linspace(first_slice, first_slice+slices-1, slices, dtype='int')
    
    for i, sl in enumerate(selected_slices):
        # 1st column: input values
        axs[i, 0].imshow(np.transpose(inputs[:,:,sl]), aspect='auto',
                         cmap='gray', vmin=min_input, vmax=max_input)
        plt.sca(axs[i, 0])
        plt.yticks(fontsize=fontsize)

        # 2nd column: ground truth
        axs[i, 1].imshow(np.transpose(ground_truth[:,:,sl]), aspect='auto',
                         cmap='turbo',vmin=min_output, vmax=max_output)

        # 3rd column: model prediction
        cbh = axs[i, 2].imshow(np.transpose(outputs[:,:,sl]), aspect='auto', 
                               cmap='turbo',vmin=min_output, vmax=max_output)

        # 4th column: difference or gamma analysis results
        if gamma_evaluation:
            axes = (np.arange(ground_truth.shape[0])*resolution[0],
                np.arange(ground_truth.shape[1])*resolution[1],
                np.arange(ground_truth.shape[2])*resolution[2])
            gamma_values = np.nan_to_num(gamma(
                axes, ground_truth, axes, outputs, 2, 2,
                lower_percent_dose_cutoff=gamma_cutoff, quiet=True), 0)
            cbh2 = axs[i, 3].imshow(np.transpose(np.absolute(gamma_values[:,:,sl])),
                aspect='auto', vmin=0, vmax=2, cmap='RdBu')
        else:
            axs[i, 3].imshow(np.transpose(np.absolute(
                ground_truth[:,:,sl]-outputs[:,:,sl])),aspect='auto',
                cmap='turbo',vmin=min_output, vmax=max_output)
    
    # Axes labels and colorbar
    axs[7, 0].set_ylabel("X-axis values [mm]", fontsize=fontsize)
    fig.text(0.435, 0.08, "Depth Y-axis values [mm]", ha='center', fontsize=fontsize)
    for j in range(num_cols):
        plt.sca(axs[i, j])
        plt.xticks(fontsize=fontsize)
    plt.subplots_adjust(hspace=0, wspace=0.05)
    cb = fig.colorbar(cbh, ax=axs, location='right')
    cb.ax.set_ylabel("[Gy/MU]", size=fontsize)
    cb.ax.tick_params(labelsize=fontsize)
    if gamma_evaluation:
        cb2 = fig.colorbar(cbh2, ax=axs, location='right')
        cb2.ax.set_ylabel("Gamma value", size=fontsize)
        cb2.ax.tick_params(labelsize=fontsize)
    plt.show()

def plot_slice(inputs, ground_truth, outputs, scale, dose_threshold=1,
    distance_threshold=3, slice_number=None, cutoff=0, figsize=(4,14), fontsize=14,
    resolution=[2.5,2.5,2.5], gamma_slice=True, savefig=True):
    """
    Plots slices of the full beam along the Z axis.
    *inputs..........3D array [Y,X,Z] from function infer
    """
    # Initialize figure and axes.
    fig, axs = plt.subplots(4, 1, figsize=figsize)
    axs[0].set_title("CT scan", fontsize=fontsize, fontweight='bold')
    axs[1].set_title("Target", fontsize=fontsize, fontweight='bold')
    axs[2].set_title("Prediction", fontsize=fontsize, fontweight='bold')
    if gamma_slice:
        axs[3].set_title("Gamma analysis", fontsize=fontsize, fontweight='bold')
    else:
        axs[3].set_title("Difference", fontsize=fontsize, fontweight='bold')
    plt.subplots_adjust(hspace=0.3, wspace=0.03)

    # Cut off MC noise
    ground_truth[ground_truth<(cutoff/100)*scale['y_max']] = 0
    outputs[outputs<(cutoff/100)*scale['y_max']] = 0

    # Calculate maximum and minimum per column.
    min_input, max_input = np.min(inputs), np.max(inputs)
    min_output, max_output = np.min(outputs), np.max(outputs)
    if slice_number is None:
        slice_number = int(np.floor(ground_truth.shape[-1]/2))
    
    # 1st row: input values
    cbh0 = axs[0].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', vmin=min_input, vmax=max_input)
    plt.sca(axs[0])
    plt.xticks([20, 40, 60, 80], ['50', '100', '150', '200'], fontsize=fontsize)
    plt.yticks([80, 60, 40, 20], ['50', '100', '150', '200'], fontsize=fontsize)
    axs[0].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[0].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb0 = fig.colorbar(cbh0, ax=axs[0], aspect=fontsize)
    cb0.ax.set_ylabel("HU [norm.]", size=fontsize)
    cb0.ax.tick_params(labelsize=fontsize)

    # 2nd row: ground truth
    axs[1].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    cbh1 = axs[1].imshow(np.transpose(ground_truth[:,:,slice_number]), aspect='auto',
        cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
    plt.sca(axs[1])
    plt.xticks([20, 40, 60, 80], ['50', '100', '150', '200'], fontsize=fontsize)
    plt.yticks([80, 60, 40, 20], ['50', '100', '150', '200'], fontsize=fontsize)
    axs[1].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[1].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb1 = fig.colorbar(cbh1, ax=axs[1], aspect=fontsize)
    cb1.ax.set_ylabel(r"Gy/MU", size=fontsize)
    cb1.ax.tick_params(labelsize=fontsize)

    # 3rd row: model prediction
    axs[2].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
        cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
    cbh2 = axs[2].imshow(np.transpose(outputs[:,:,slice_number]), aspect='auto', 
        cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
    plt.sca(axs[2])
    plt.xticks([20, 40, 60, 80], ['50', '100', '150', '200'], fontsize=fontsize)
    plt.yticks([80, 60, 40, 20], ['50', '100', '150', '200'], fontsize=fontsize)
    axs[2].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[2].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb2 = fig.colorbar(cbh2, ax=axs[2], aspect=fontsize)
    cb2.ax.set_ylabel(r"Gy/MU", size=fontsize)
    cb2.ax.tick_params(labelsize=fontsize)

    # 4th row: difference or gamma analysis results
    if gamma_slice:
        axes = (np.arange(ground_truth.shape[0])*resolution[0],
            np.arange(ground_truth.shape[1])*resolution[1],
            np.arange(ground_truth.shape[2])*resolution[2])
        gamma_values = np.nan_to_num(
            gamma(axes, ground_truth, axes, outputs, dose_threshold,
            distance_threshold, lower_percent_dose_cutoff=0.1, quiet=True), 0)
        axs[3].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
        cbh3 = axs[3].imshow(np.transpose(np.absolute(gamma_values[:,:,slice_number])),
            aspect='auto', alpha=0.6, vmin=0, vmax=2, cmap='RdBu')

    else:
        axs[3].imshow(np.transpose(inputs[:,:,slice_number]), aspect='auto',
            cmap='gray', alpha=0.4, vmin=min_input, vmax=max_input)
        cbh3 = axs[3].imshow(np.transpose(np.absolute(
            ground_truth[:,:,slice_number]-outputs[:,:,slice_number])),
            aspect='auto', cmap='turbo', alpha=0.6, vmin=min_output, vmax=max_output)
        
    plt.sca(axs[3])
    plt.xticks([20, 40, 60, 80], ['50', '100', '150', '200'], fontsize=fontsize)
    plt.yticks([80, 60, 40, 20], ['50', '100', '150', '200'], fontsize=fontsize)
    axs[3].set_ylabel("mm", loc='top', fontsize=fontsize)
    axs[3].set_xlabel("mm", loc='right', fontsize=fontsize)
    cb3 = fig.colorbar(cbh3, ax=axs[3], aspect=fontsize)
    if gamma_slice:
        cb3.ax.set_ylabel(r"$\gamma$ value", size=fontsize)
    else:
        cb3.ax.set_ylabel(r"Gy/MU", size=fontsize)
    cb3.ax.tick_params(labelsize=fontsize)
        
    if savefig:
        plt.savefig(time.strftime('%Y%m%d-%H%M'), dpi=300, bbox_inches='tight') 

    plt.show()

def plot_coronal(inputs, ground_truth, outputs, scale, slice_number, n_slices=5,
    cutoff=0, figsize=(15,8), fontsize=15, resolution=[2,2,2], savefig=True):
    """
    Plots slices of the full beam along the Z axis.
    *inputs..........3D array [Y,X,Z] from function infer
    """
    # Initialize figure and axes.
    fig, axs = plt.subplots(4, n_slices, figsize=figsize)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # Cut off MC noise
    ground_truth[ground_truth<(cutoff/100)*scale['y_max']] = 0
    outputs[outputs<(cutoff/100)*scale['y_max']] = 0

    # Calculate maximum and minimum per column.
    min_input, max_input = np.min(inputs), np.max(inputs)
    min_output, max_output = np.min(outputs), np.max(outputs)

    for i in range(n_slices):
    
        # 1st row: input values
        cbh0 = axs[0,i].imshow(np.transpose(inputs[slice_number+i,:,:]),
            aspect='equal', cmap='gray', vmin=min_input, vmax=max_input)
        axs[0,i].set_xticklabels([])
        axs[0,i].set_yticklabels([])
        plt.sca(axs[0,i])
        if i is n_slices-1:
            cb0 = fig.colorbar(cbh0, ax=axs[0,], aspect=fontsize)
            cb0.ax.set_ylabel("HU", size=fontsize)
            cb0.ax.tick_params(labelsize=fontsize*0.75)
        if i == 0:
            axs[0,i].set_ylabel("CT scan", fontsize=fontsize)

        # 2nd row: ground truth
        cbh1 = axs[1,i].imshow(np.transpose(ground_truth[slice_number+i,:,:]),
            aspect='equal', cmap='turbo', vmin=min_output, vmax=max_output)
        plt.sca(axs[1,i])
        axs[1,i].set_xticklabels([])
        axs[1,i].set_yticklabels([])
        if i is n_slices-1:
            cb1 = fig.colorbar(cbh1, ax=axs[1:,], aspect=fontsize*2.5)
            cb1.ax.set_ylabel(r"Gy/$10^9$ particles", size=fontsize)
            cb1.ax.tick_params(labelsize=fontsize*0.75)
        if i == 0:
            axs[1,i].set_ylabel("MC dose", fontsize=fontsize)

        # 3rd row: model prediction
        cbh2 = axs[2,i].imshow(np.transpose(outputs[slice_number+i,:,:]), 
            aspect='equal', cmap='turbo', vmin=min_output, vmax=max_output)
        plt.sca(axs[2,i])
        axs[2,i].set_xticklabels([])
        axs[2,i].set_yticklabels([])
        if i == 0:
            axs[2,i].set_ylabel("Prediction", fontsize=fontsize)


        # 4th row: dose difference 
        cbh3 = axs[3,i].imshow(np.transpose(np.absolute(
            ground_truth[slice_number+i,:,:]-outputs[slice_number+i,:,:])),
            aspect='equal', cmap='turbo', vmin=min_output, vmax=max_output)
        axs[3,i].set_xticklabels([])
        axs[3,i].set_yticklabels([])
        if i == 0:
            axs[3,i].set_ylabel("Difference", fontsize=fontsize)
        
    if savefig:
        plt.savefig(time.strftime('%Y%m%d-%H%M'), dpi=300, bbox_inches='tight') 

    plt.show()