import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from spikingjelly.datasets import dvs128_gesture
from fxpmath import Fxp


def tensor_is_raw(tensor, bin_rep=False):
    """
    Used when the tensor is the raw data from the dataloader. 
    Expects a shape of (16,2,128,128), AKA one tensor from the batch, not scaled by 3/max(x) yet. 
    Outputs the "quantized" tensor of shape (16,2,64,64).
    """
    x = Q3_5QuantizedTensor()

    if tensor.ndimension() == 5:
        tensor = tensor.view(-1, *tensor.shape[2:])

    tensor = 3*tensor/(torch.max(tensor))
    tensor = torch.nn.functional.max_pool2d(tensor, kernel_size=2, stride=2)
    #tensor = quantize_tensor(tensor, bin_rep=bin_rep)
    tensor = x.quantize(tensor)
    tensor = tensor.view(-1, 16, 2, 64, 64)

    return tensor

def quantize_tensor_for_coe(x, return_fxp=False, bin_rep=False, n_word=8, n_frac=5):
    base = Fxp(None, signed=True, n_word=n_word, n_frac=n_frac, rounding="floor")
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


class Q3_5QuantizedTensor:
    """Helper class to simulate Q3.5 fixed-point format"""
    def __init__(self, tensor=None, n_word=8, n_frac=5):    
        # Q3.5 parameters
        self.int_bits = n_word - n_frac  # 3 bits for integer including sign bit
        self.frac_bits = n_frac  # 5 bits for fraction
        self.scale = 2 ** self.frac_bits  # 32
        self.min_val = -2 ** (self.int_bits - 1)  # -4
        self.max_val = 2 ** (self.int_bits - 1) - 2 ** (-self.frac_bits)  # ~3.97
        
        # Store the quantized tensor
        self.tensor = None
        if tensor is not None:
            self.tensor = self.quantize(tensor)
        self.n_word = n_word
        self.n_frac = n_frac
    
    def quantize(self, tensor):
        """Quantize a floating-point tensor to Q3.5"""
        # Convert to fixed-point representation
        fixed_point = torch.floor(tensor * self.scale)
        
        # Clamp to valid Q3.5 range
        #fixed_point = torch.clamp(fixed_point, 
        #                         self.min_val * self.scale, 
        #                         self.max_val * self.scale)
        
        # Convert back to floating-point with Q3.5 precision
        return fixed_point / self.scale


class Q3_5Hook:
    """Hook to quantize activations to Q3.5 format"""
    def __init__(self, n_word=8, n_frac=5):
        self.q3_5 = Q3_5QuantizedTensor(n_frac=n_frac, n_word=n_word)
    
    def __call__(self, module, input_tensor, output_tensor):
        # Quantize the output to Q3.5 format
        return self.q3_5.quantize(output_tensor)


def create_coe_file(tensor, filename, radix=2, convert_to_bin=False):
    """
    Given a tensor, create a .coe file with the values in the tensor. 
    """
    if convert_to_bin:
        tensor = quantize_tensor_for_coe(tensor, bin_rep=True)

    with open(filename, 'w') as f:
        # Write the radix header
        f.write(f"memory_initialization_radix={radix};\n")
        f.write("memory_initialization_vector=\n")
        
        # Let qunatize weights take care of fixed point formatting
        if (radix == 2):
            values = [f"{int(str(int(val)), 2) & 0xFF:08b}" for val in tensor.flatten()]
        else:
            values = [f"{int(val.item())}" for val in tensor.flatten()]

        # Write values as a comma-separated list
        f.write(",\n".join(values) + ";\n")

def read_coe(file, in_bin=False):
    """
    Read a .coe file and return the values as a list of floats.
    """
    base = Fxp(None, signed=True, n_word=8, n_frac=5)
    with open(file, 'r') as f:
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

def plot_tensor_distribution(x):
    # Convert tensor to numpy
    x = x.numpy()

    # Define bins
    # Create logarithmically spaced bins from 0.01 to 100
    # define bins with linear spacing
    start = 1e-5
    stop = 1e-1
    num_bins = 10
    step = (stop - start) / num_bins
    edges = np.arange(start, stop + step, step)
    
    # Add -inf and +inf for overflow bins
    bins = [-float('inf')] + edges.tolist() + [float('inf')]

    # Create readable labels
    labels = [f"<{start}"] + [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)] + [f">{stop}"]

    # digitize the data into the bins

    bin_indices = np.digitize(np.abs(x), bins) - 1

    # Count occurrences in each bin
    counts = [np.sum(bin_indices == i) for i in range(len(bins) - 1)]

    # Plot histogram
    plt.bar(labels, counts)
    plt.xlabel('Value Range')
    plt.ylabel('Count')
    plt.title('Distribution of Tensor Values')
    plt.show()

"""
Common statements
"""

device = torch.device("cuda")
seed = 19
generator = torch.Generator().manual_seed(seed)
torch.set_printoptions(threshold=float("inf"), precision=10)
torch.set_grad_enabled(False)

# Dataset Directory
DVSGesture_dir = "/home/aaron/Research_Projs/DvsGesture/"
root_dir  = os.path.join(DVSGesture_dir, "events_np/train/0")
frame_dir = os.path.join(DVSGesture_dir, "frames_number_16_split_by_number/train/0")

# Working Directory
working_dir = "/home/aaron/Research_Projs/Spike-Element-Wise-ResNet/"
checkpoint_path = os.path.join(working_dir, "dvsgesture/logs/26_no_bias/lr0.001")
checkpoint_file = "checkpoint_299.pth"

model_params = os.path.join(checkpoint_path, checkpoint_file)

# Dataset
dataset_train = dvs128_gesture.DVS128Gesture(root=DVSGesture_dir, train=True, data_type='frame', frames_number=16, split_by='number')
sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=generator)

dataset_test  = dvs128_gesture.DVS128Gesture(root=DVSGesture_dir, train=False, data_type='frame', frames_number=16, split_by='number')
sampler_test = torch.utils.data.RandomSampler(dataset_test, generator=generator)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=49,
    sampler=sampler_train, num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=32,
    sampler=sampler_test, num_workers=0, pin_memory=True)

# coe file related
test_data_coe = "./coe_files/test_data.coe"
conv_coe = "./coe_files/conv3x3.coe"
fc_data_coe = [f"./coe_files/fc{i}.coe" for i in range(11)]