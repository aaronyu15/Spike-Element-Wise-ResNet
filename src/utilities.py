import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from spikingjelly.datasets import dvs128_gesture
from fxpmath import Fxp
from torchvision import transforms
import re
from collections import defaultdict
import pandas as pd


def data_preprocess(tensor, bin_rep=False, args=None):
    """
    Used when the tensor is the raw data from the dataloader. 
    Expects a shape of (16,2,128,128), AKA one tensor from the batch, not scaled by 3/max(x) yet. 
    Outputs the "quantized" tensor of shape (16,2,64,64).
    """
    x = Q3_5QuantizedTensor()

    N, B, C, H, W = tensor.shape

    if tensor.ndimension() == 5:
        tensor = tensor.view(-1, *tensor.shape[2:])

    # Scale tensor values to be between 0 and 3, and then apply max pooling
    tensor = 3*tensor/(torch.max(tensor))
    tensor = torch.nn.functional.max_pool2d(tensor, kernel_size=2, stride=2)
    if(args.use_coe):
        tensor = x.quantize(tensor)
    tensor = tensor.view(N, B, C, H//2, W//2)

    return tensor

def quantize_tensor_for_coe(x, return_fxp=False, bin_rep=False, n_word=8, n_frac=5):
    #base = Fxp(None, signed=True, n_word=n_word, n_frac=n_frac, rounding='floor') Remove round to match coe files
    base = Fxp(None, signed=True, n_word=n_word, n_frac=n_frac )
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
    x = x.detach().numpy()

    # Define bins
    # Create logarithmically spaced bins from 0.01 to 100
    # define bins with linear spacing
    start = 1e-4
    stop = 1e1
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

def play(tensor, save_gif=False, file_name="wave.gif"):
    to_img = transforms.ToPILImage()
    img_tensor = torch.zeros([tensor.shape[0], 3, tensor.shape[2], tensor.shape[3]])
    img_tensor[:, 1] = tensor[:, 0]
    img_tensor[:, 2] = tensor[:, 1]
    for t in range(img_tensor.shape[0]):
            plt.imshow(to_img(img_tensor[t]))
            plt.pause(0.2)

    if save_gif:
        frames = []
        for t in range(img_tensor.shape[0]):
            img = to_img(img_tensor[t])  # Convert tensor to PIL image
            frames.append(img.copy())  # Make a copy so we don’t overwrite

        # Save the frames as an animated GIF
        frames[0].save(
            file_name,
            save_all=True,
            append_images=frames[1:],
            duration=int(0.2 * 1000),  # duration in milliseconds
            loop=0
        )
    print(f"Saved animation to {file_name}")

# Dataset
# Define custom transform for salt and pepper noise
class SaltPepperNoise:
    def __init__(self, prob=0.05):
        self.prob = prob

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        min_val = torch.amin(img, dim=(0, 2, 3), keepdim=True)  # Shape: (1, 1, C, 1, 1)
        max_val = torch.amax(img, dim=(0, 2, 3), keepdim=True)  # Shape: (1, 1, C, 1, 1)

        # Most entries in the tensor are less than 10. The maximum is typically near or over 100. 
        # Using max_val/10 is suitable to represent a possible noise value.
        # min_val is always 0 but included for completeness.
        noise = torch.rand_like(img)
        img = torch.where(noise < self.prob / 2, min_val/10, img)
        img = torch.where(noise > 1 - self.prob / 2, max_val/10, img)
        return img


"""

    Parse batch_output.log to extract Top-1 and Top-3 accuracies for each parameter set.
    
"""
def parse_batch_log(file_path):
    # Dictionary to store accuracies for each parameter set
    accuracies = defaultdict(lambda: {'top1': [], 'top3': []})
    current_param = None
    
    # Regular expressions to match accuracy lines
    top1_pattern = re.compile(r'Top-1 Accuracy on the test dataset: (\d+\.\d+)%')
    top3_pattern = re.compile(r'Top-3 Accuracy on the test dataset: (\d+\.\d+)%')
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Identify parameter set
        if line.startswith('Processing parameter:'):
            current_param = line.split('Processing parameter: ')[1].strip()
        
        # Extract Top-1 Accuracy
        top1_match = top1_pattern.search(line)
        if top1_match and current_param:
            accuracies[current_param]['top1'].append(float(top1_match.group(1)))
        
        # Extract Top-3 Accuracy
        top3_match = top3_pattern.search(line)
        if top3_match and current_param:
            accuracies[current_param]['top3'].append(float(top3_match.group(1)))
    
    return accuracies

def calculate_averages(accuracies):
    # Calculate average Top-1 and Top-3 accuracies for each parameter set
    results = []
    for param, acc in accuracies.items():
        avg_top1 = sum(acc['top1']) / len(acc['top1']) if acc['top1'] else 0
        avg_top3 = sum(acc['top3']) / len(acc['top3']) if acc['top3'] else 0
        results.append({
            'Parameter': param,
            'Avg Top-1 Accuracy (%)': round(avg_top1, 2),
            'Avg Top-3 Accuracy (%)': round(avg_top3, 2)
        })
    return results

def main():
    log_file = 'batch_output.log'
    csv_file =  log_file.replace('.log', '.csv')

    # Parse the log file
    accuracies = parse_batch_log(log_file)
    
    # Calculate averages
    results = calculate_averages(accuracies)
    
    # Convert to DataFrame for nice formatting
    df = pd.DataFrame(results)
    
    # Print results
    print("\nAverage Top-1 and Top-3 Accuracies per Parameter Set:")
    print(df.to_string(index=False))
    
    # Optionally save to CSV
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

if __name__ == "__main__":
    main()
    print("Batch processing completed.")