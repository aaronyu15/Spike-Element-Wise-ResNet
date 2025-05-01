from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.datasets import dvs128_gesture
import custom_spikingjelly_class
import numpy as np
import os
import torch
import dvsgesture.smodels as smodels
from fxpmath import Fxp

"""
HW_output_test.py

This script processes COE files to generate tensors, performs convolutional 
operations, applies a custom spiking neural network layer, and generates 
predictions using a fully connected layer. It is part of the Spike-Element-Wise-ResNet 
project and is designed to test hardware output.

Key functionalities:
- Reads and processes COE files into tensors.
- Performs convolution, spiking neural network operations, and pooling.
- Generates predictions using a fully connected layer.
"""

def create_flat_tensor_from_file(file_path):
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Memory" in line:
                # Extract the value after 'Memory[x] ='
                value = int(line.split('=')[-1].strip())
                values.append(value)
    
    # Create a flat tensor from the extracted values
    tensor = torch.tensor(values, dtype=torch.int32)
    return tensor

def quantize_weights(x, return_fxp=False, bin_rep=False):
    base = Fxp(None, signed=True, n_word=8, n_frac=5, rounding="floor")
    if isinstance(x, int) or isinstance(x, float):
        if(return_fxp):
            q = base(x)
        else:
            q = float(base(x))
    elif torch.is_tensor(x):
        q = torch.empty_like(x, dtype=torch.float32)
        for i in range(x.numel()):
            val = x.flatten()[i].item()
            if(bin_rep):
                val = float(base(val).bin())
            else:
                val = float(base(val))

            q.flatten()[i] = val

    return q

base = Fxp('0b11111000', signed=True, n_word=8, n_frac=5, rounding="floor")

device = torch.device("cuda")
seed = 18
generator = torch.Generator().manual_seed(seed)
torch.set_printoptions(threshold=float("inf"), precision=5)
torch.set_grad_enabled(False)



dummydata = "./coe_files/dummy.coe"
conv = "./coe_files/conv3x3.coe"
fcdata = "./coe_files/fc{}.coe"

def read_coe(fa, in_bin=False):
    base = Fxp(None, signed=True, n_word=8, n_frac=5)
    with open(fa, 'r') as f:
        lines = f.readlines()

    data = []
    for i, line in enumerate(lines):
        if i < 2:
            continue
        # Remove commas and whitespace, convert to integer
        cleaned = line.strip().replace(',', '').replace(';', '')
        if cleaned.isdigit():
          if(in_bin):
            data.append(float(base('0b' + cleaned)))
          else:
            data.append(float(cleaned))
    return data

image_data = read_coe(dummydata, in_bin=True)
conv_data = read_coe(conv, in_bin=True)
fc = []

for i in range(11):
    fc.append(read_coe(fcdata.format(i), in_bin=True))

    


T, C, H, W = 16, 2, 64, 64

image = torch.tensor(image_data).reshape(T, C, H, W)
conv_data_weight = torch.tensor(conv_data[0:-6])
conv_data_bias = torch.tensor(conv_data[-6:])
fc = torch.tensor(fc)



# Convolution
c = torch.nn.Conv2d(in_channels=2, out_channels=6, kernel_size=3, padding=0, bias=True)
conv_data_weight = conv_data_weight.reshape(6,2,3,3)
c.weight.data = conv_data_weight
c.bias.data = conv_data_bias
res = c(image)
# LIF
n = custom_spikingjelly_class.custom_MultiStepParametricLIFNode(init_tau=0.5, decay_input=True, v_threshold=1.0)
res = n(res)
# MP
res = torch.nn.functional.max_pool2d(res, kernel_size=2, stride=2)
res = res.mean(0)
res = res.flatten()
# Fully conencted
f = torch.nn.Linear(in_features=6 * 31 *31, out_features=11, bias=False)
f.weight.data = fc

res = torch.nn.functional.softmax(f(res), dim=0)
print(res)
print(torch.topk(res, 3))



exit()
