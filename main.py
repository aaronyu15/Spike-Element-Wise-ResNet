import struct
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.datasets import dvs128_gesture
import custom_spikingjelly_class
import numpy as np
import os
import torch
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
import dvsgesture.smodels as smodels
from fxpmath import Fxp
import torchvision
import utilities
from utilities import data_loader, test_loader, model_params, device

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
    if(isinstance(sequential, smodels.SEWBlock)):
        x = sequential.conv
    else:
        x = sequential

    for layer in x:
        if isinstance(layer, nn.Sequential) or isinstance(layer, smodels.SEWBlock):
            #layers.extend(flatten_sequential(layer))  # Recursively flatten
            layers.append(layer)
        else:
            layers.append(layer)
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





def play(x):
    #to_img = transforms.ToPILImage()
    #img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    #img_tensor[:, 1] = x[:, 0]
    #img_tensor[:, 2] = x[:, 1]
    #for i in range(2):
    #    for t in range(img_tensor.shape[0]):
    #            plt.imshow(to_img(img_tensor[t]))
    #            plt.pause(0.2)
    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([x.shape[0], 3, x.shape[2], x.shape[3]])
    img_tensor[:, 1] = x[:, 0]  # Green channel
    img_tensor[:, 2] = x[:, 1]  # Blue channel

    frames = []

    for t in range(img_tensor.shape[0]):
        img = to_img(img_tensor[t])  # Convert tensor to PIL image
        frames.append(img.copy())  # Make a copy so we don’t overwrite

    # Save the frames as an animated GIF
    frames[0].save(
        "wave.gif",
        save_all=True,
        append_images=frames[1:],
        duration=int(0.2 * 1000),  # duration in milliseconds
        loop=0
    )

    print(f"Saved animation to {'wave.gif'}")


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.

    Args:
    - model: The trained model to evaluate.
    - test_loader: DataLoader for the test dataset.
    - device: The device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
    - accuracy: The accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test dataset: {accuracy:.2f}%")
    return accuracy


# Software model
checkpoint = torch.load(model_params, weights_only=False)

model = smodels.ResNetN()
model.load_state_dict(checkpoint["model"])
model = model.to("cuda:0")
model.eval()

# Call the evaluation function
evaluate_model(model, test_loader, device)

# Data classes are 0 - 10, not 1 - 11

exit()

# Flatten the model for printing
flat_model = flatten_sequential(model.children())


# First data of size (16, 2, 128, 128) for testing
data_iter = iter(data_loader)
test_iter = iter(test_loader)
first_data = next(test_iter)
input_data, label = first_data


input_data = input_data.squeeze()
play(input_data)

exit()

#generate_dummy_coe(input_data)




def create_model_coe_files():
    bin_rep = False
    tw1, tb1 = fold_batchnorm_no_bias(flat_model[1][0].weight,  flat_model[2][0].weight, flat_model[2][0].bias, flat_model[2][0].running_mean, flat_model[2][0].running_var, flat_model[2][0].eps)
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

    a = torch.cat((
        conv3x3coe,
        fc0 ,
        fc1 ,
        fc2 ,
        fc3 ,
        fc4 ,
        fc5 ,
        fc6 ,
        fc7 ,
        fc8 ,
        fc9 ,
        fc10, 
    ), dim=0)

    print(a.shape)

    utilities.plot_tensor_distribution(a)

    conv3x3coe_q = utilities.quantize_weights(conv3x3coe, bin_rep=bin_rep)
    fc0_q  = utilities.quantize_weights(fc0 , bin_rep=bin_rep)
    fc1_q  = utilities.quantize_weights(fc1 , bin_rep=bin_rep)
    fc2_q  = utilities.quantize_weights(fc2 , bin_rep=bin_rep)
    fc3_q  = utilities.quantize_weights(fc3 , bin_rep=bin_rep)
    fc4_q  = utilities.quantize_weights(fc4 , bin_rep=bin_rep)
    fc5_q  = utilities.quantize_weights(fc5 , bin_rep=bin_rep)
    fc6_q  = utilities.quantize_weights(fc6 , bin_rep=bin_rep)
    fc7_q  = utilities.quantize_weights(fc7 , bin_rep=bin_rep)
    fc8_q  = utilities.quantize_weights(fc8 , bin_rep=bin_rep)
    fc9_q  = utilities.quantize_weights(fc9 , bin_rep=bin_rep)
    fc10_q = utilities.quantize_weights(fc10, bin_rep=bin_rep)



    #utilities.create_coe_file(conv3x3coe_q, filename=f"coe_files/conv3x3.coe")
    #utilities.create_coe_file(fc0_q, filename=f"coe_files/fc0.coe")
    #utilities.create_coe_file(fc1_q , filename=f"coe_files/fc1.coe")
    #utilities.create_coe_file(fc2_q , filename=f"coe_files/fc2.coe")
    #utilities.create_coe_file(fc3_q , filename=f"coe_files/fc3.coe")
    #utilities.create_coe_file(fc4_q , filename=f"coe_files/fc4.coe")
    #utilities.create_coe_file(fc5_q , filename=f"coe_files/fc5.coe")
    #utilities.create_coe_file(fc6_q , filename=f"coe_files/fc6.coe")
    #utilities.create_coe_file(fc7_q , filename=f"coe_files/fc7.coe")
    #utilities.create_coe_file(fc8_q , filename=f"coe_files/fc8.coe")
    #utilities.create_coe_file(fc9_q , filename=f"coe_files/fc9.coe")
    #utilities.create_coe_file(fc10_q, filename=f"coe_files/fc10.coe")


create_model_coe_files()




def inspect_model_layers(x, data):
    # Input stage / layers
    data = data.permute(1, 0, 2, 3, 4)

    x = flat_model[0](data)
    x = flat_model[1](x)
    print(f"Conv2d: {x[0,0,0,0]}")

    print(x.shape)
    x = flat_model[2](x)
    print(f"batchnorm: {x[0,0,0,0]}")

    x = flat_model[3](x)
    print(f"LIF: {x[0,0,0,0]}")

    x = flat_model[4](x)
    print(f"pool: {x[0,0,0,0]}")

    x = flat_model[5](x)
    print(f"flatten: {x.shape}")
    for i in range(16):
        print(f"{i} : {x[i,0,0]}")


    x = x.mean(0)
    print(f"mean 0: {x.shape}")
    print(f"0: {x[0,0]}")

#inspect_model_layers(x, input_data)







