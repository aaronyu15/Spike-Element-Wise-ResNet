import struct
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
import spikingjelly.clock_driven.layer as sp_layer
from spikingjelly.datasets import dvs128_gesture
import custom_spikingjelly_class
import numpy as np
import os
import torch
from torch import nn
from matplotlib import pyplot as plt
from fxpmath import Fxp
import torchvision
import utilities
import sys
import Model
"""
main.py

This script is part of the Spike-Element-Wise-ResNet project. It includes 
utilities for processing spiking neural network models, generating COE files, 
and visualizing data. The script also provides functions for folding BatchNorm 
parameters, quantizing weights, and inspecting model layers.

Key functionalities:
- Processes DVS128 Gesture dataset for training and testing.
- Implements functions to fold BatchNorm parameters into Conv2D weights.
- Quantizes weights and generates COE files for hardware compatibility.
- Visualizes tensor distributions and plays input data as animations.
- Provides utilities for inspecting and debugging model layers.
"""

def flatten_sequential(sequential):
    """
    Flatten the model into a single list for inspection
    """
    layers = []

    for layer in sequential.children():
        # Recursively flatten containers, but only collect foundational layers (nn.Conv2d, nn.Linear, etc.)
        if isinstance(layer, (nn.Sequential, sp_layer.SeqToANNContainer)):
            layers.extend(flatten_sequential(layer))
        else:
            layers.append(layer)
        # Optionally, add more foundational types as needed
    return layers

def fold_batchnorm_no_bias(conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_eps):
    """
    Folds BatchNorm parameters into Conv2D weights when Conv2D has no bias.
    
    Args:
    conv_weight: Tensor of shape (out_channels, in_channels, kH, kW) - Conv2D weights
    bn_weight: Tensor of shape (out_channels,) - BatchNorm gamma (scale)
    bn_bias: Tensor of shape (out_channels,) - BatchNorm beta (shift)
    bn_running_mean: Tensor of shape (out_channels,) - BatchNorm running mean
    bn_running_var: Tensor of shape (out_channels,) - BatchNorm running variance
    bn_eps: float - BatchNorm epsilon

    Returns:
    new_weight: Tensor of same shape as conv_weight
    new_bias: Tensor of shape (out_channels,)
    """
    # Compute scaling factor for each output channel
    scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
    
    # Scale the convolutional weights
    new_weight = conv_weight * scale.view(-1, 1, 1, 1)

    # Compute new bias (since Conv2D has no bias, it starts at zero)
    new_bias = bn_bias - scale * bn_running_mean
    
    return new_weight, new_bias

def create_model_coe_files(flat_model=None, bin_rep=False):
    save_dir = "../coe_files_tmp"
    radix = 10
    log_values = True
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tw1, tb1 = fold_batchnorm_no_bias(flat_model[1].weight,
                                      flat_model[2].weight,
                                      flat_model[2].bias,
                                      flat_model[2].running_mean,
                                      flat_model[2].running_var,
                                      flat_model[2].eps)

    tw1 = tw1.flatten()
    tb1 = tb1.flatten()
    conv3x3coe = torch.cat((tw1, tb1), dim=0)

    fc0  = flat_model[6].weight[0] 
    fc1  = flat_model[6].weight[1] 
    fc2  = flat_model[6].weight[2] 
    fc3  = flat_model[6].weight[3] 
    fc4  = flat_model[6].weight[4] 
    fc5  = flat_model[6].weight[5] 
    fc6  = flat_model[6].weight[6] 
    fc7  = flat_model[6].weight[7] 
    fc8  = flat_model[6].weight[8] 
    fc9  = flat_model[6].weight[9] 
    fc10 = flat_model[6].weight[10] 

    wr_f = ["conv3x3", "fc0", "fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "fc7", "fc8", "fc9", "fc10"]
    wr   = [conv3x3coe, fc0, fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10]

    if log_values:

        for i, write_file in enumerate(wr_f):
            path = os.path.join(save_dir, f"{write_file}.txt")
            with open(path, 'w') as f:
                tensor = wr[i].flatten()
                for val in tensor:
                    val = float(val)
                    f.write(f"{val}\n")


    for i, write_file in enumerate(wr_f):
        path = os.path.join(save_dir, f"{write_file}.coe")
        utilities.create_coe_file(wr[i], filename=path, radix=radix, convert_to_bin=bin_rep)


def reverse_coe_files():
    read_dir = "../coe_files_tmp"
    save_dir = "../coe_files_new_tmp"
   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    conv_coe_f = "conv3x3"
    fc0_f  =  "fc0"
    fc1_f  =  "fc1"
    fc2_f  =  "fc2"
    fc3_f  =  "fc3"
    fc4_f  =  "fc4"
    fc5_f  =  "fc5"
    fc6_f  =  "fc6"
    fc7_f  =  "fc7"
    fc8_f  =  "fc8"
    fc9_f  =  "fc9"
    fc10_f =  "fc10"

    l = [conv_coe_f, fc0_f, fc1_f, fc2_f, fc3_f, fc4_f, fc5_f, fc6_f, fc7_f, fc8_f, fc9_f, fc10_f]

    for fn in l:
        rd = os.path.join(read_dir, f"{fn}.coe")
        wr = os.path.join(save_dir, f"{fn}.txt")

        data = utilities.read_coe(rd, in_bin=True)       

        with open(wr, 'w') as f:
            for i in data:
                # Convert to float and write to file
                val = float(i)
                f.write(f"{val}\n")

if __name__ == "__main__":

    checkpoint_path = os.path.join(".", "../dvsgesture/logs/26_no_bias/lr0.001")
    checkpoint_file = "checkpoint_299.pth"

    model_params = os.path.join(checkpoint_path, checkpoint_file)

    # Software model
    checkpoint = torch.load(model_params, weights_only=False)
    
    model = Model.ResNetN()
    model.load_state_dict(checkpoint["model"])
    
    # Data classes are 0 - 10, not 1 - 11
    # Load the model and flatten it
    flat_model = flatten_sequential(model)

    
    # Create COE files for the model weights
    create_model_coe_files(flat_model, bin_rep=True)

    reverse_coe_files()

    # Optionally, you can also visualize the model layers or perform other operations
    # utilities.visualize_model_layers(flat_model)  # Uncomment if you have a visualization function










