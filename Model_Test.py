import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import layer
from spikingjelly.datasets import dvs128_gesture
import custom_spikingjelly_class
import os
import utilities
from utilities import data_loader, test_loader, model_params, device, Q3_5QuantizedTensor, Q3_5Hook
import argparse
import dvsgesture.smodels as smodels

class ResNetN(nn.Module):
    def __init__(self, num_classes=11, args=None):
        super().__init__()
        self.args = args

        self.maxpool0 = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        if(args.use_coe):
            self.conv2d_1 = layer.SeqToANNContainer(nn.Conv2d(2, 6, kernel_size=3, padding=0, stride=1, bias=True))
        else:
            self.conv2d_1 = layer.SeqToANNContainer(nn.Conv2d(2, 6, kernel_size=3, padding=0, stride=1, bias=False))
            self.bn1 = layer.SeqToANNContainer(nn.BatchNorm2d(6))

        self.LIF1 = custom_spikingjelly_class.custom_MultiStepParametricLIFNode(init_tau=0.5, detach_reset=True)
        self.maxpool1 = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.flat = nn.Flatten(2)

        with torch.no_grad():
            x = torch.zeros([1, 2, 128, 128])
            for m in self.modules():
                if isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Conv2d):
                    x = m(x)
            out_features = x.numel() 

        self.out = nn.Linear(out_features, num_classes, bias=False)

        # Initialize quantization
        self.q3_5 = Q3_5QuantizedTensor()
        
        # Store hooks for later removal if needed
        self.hooks = []
    
    def apply_quantization_hooks(self):
        """Apply Q3.5 quantization hooks to all layers"""
        # Remove existing hooks first
        self.remove_quantization_hooks()
        
        # Apply hooks to all layers
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.MaxPool2d)):
                hook = module.register_forward_hook(Q3_5Hook(n_word=17, n_frac=10))
                self.hooks.append(hook)
    
    def remove_quantization_hooks(self):
        """Remove all quantization hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def quantize_parameters(self):
        """Quantize all model parameters to Q3.5 format"""
        for name, param in self.named_parameters():
            with torch.no_grad():
                param.data = self.q3_5.quantize(param.data)

    def forward(self, x: torch.Tensor):
        #if(not self.args.use_coe):
        #    with torch.no_grad():
        #        x = 3*x/(torch.max(x))

        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]

        #if(not self.args.use_coe):
        #    x = self.maxpool0(x)

        x = self.conv2d_1(x)

        if(not self.args.use_coe):
            x = self.bn1(x) 
        x = self.LIF1(x) 
        x = self.maxpool1(x) 

        x = self.flat(x)
        x = self.out(x.mean(0))

        return x


def load_and_quantize_weights(model, weights_path=None, use_coe=False):
    """
    Load pre-trained weights and quantize them to Q3.5.
    If weights_path is None, just quantize the initiated random weights.
    If use_coe is True, load weights from COE files. This more resembles the hardware implementation.
    """

    if(use_coe):
        conv_data    = utilities.read_coe(utilities.conv_coe, in_bin=True)

        fc_weight = []

        for i in range(11):
            fc_weight.append(utilities.read_coe(utilities.fc_data_coe[i], in_bin=True))

        conv_data_weight = torch.tensor(conv_data[0:-6])
        conv_data_bias = torch.tensor(conv_data[-6:])
        fc_weight = torch.tensor(fc_weight)

        conv_data_weight = conv_data_weight.reshape(6,2,3,3)
        model.conv2d_1[0].weight.data = conv_data_weight
        model.conv2d_1[0].bias.data = conv_data_bias  

        model.out.weight.data = fc_weight

    elif weights_path is not None:
        # Load pre-trained weights
        checkpoint = torch.load(weights_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])

    
    # Quantize all parameters to Q3.5
    model.quantize_parameters()
    
    return model


def test_quantized_network(args):
    """Test the quantized network with example data"""
    model = ResNetN(args=args)
    
    # Load and quantize weights
    model = load_and_quantize_weights(model, weights_path=model_params, use_coe=args.use_coe)
    model = model.to(device)

    # Make sure hooks are applied
    model.apply_quantization_hooks()


    # Initialize variables to track accuracy
    correct = 0
    top3_correct = 0

    total = 0

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data, labels in test_loader:
            # Move data and labels to the same device as the model
            data, labels = data.to(device), labels.to(device)

            # tensor_is_raw() performs the preprocessing of the data (normalization, maxpooling, quantization)
            data = utilities.tensor_is_raw(data, bin_rep=False)


            # Forward pass through the model
            outputs = model(data)

            #utilities.create_coe_file(utilities.quantize_tensor(data, bin_rep=True), filename="coe_files/test_data_compare.coe")
            #utilities.create_coe_file(model.out.weight[0], filename=f"coe_files/fc0.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[1] , filename=f"coe_files/fc1.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[2] , filename=f"coe_files/fc2.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[3] , filename=f"coe_files/fc3.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[4] , filename=f"coe_files/fc4.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[5] , filename=f"coe_files/fc5.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[6] , filename=f"coe_files/fc6.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[7] , filename=f"coe_files/fc7.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[8] , filename=f"coe_files/fc8.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[9] , filename=f"coe_files/fc9.coe" , convert_to_bin=True)
            #utilities.create_coe_file(model.out.weight[10], filename=f"coe_files/fc10.coe", convert_to_bin=True)


            # Get the predicted class (highest score)
            _, predicted = torch.max(outputs, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top3_correct += torch.sum(outputs.topk(3, dim=1)[1] == labels.view(-1, 1)).item()


    # Calculate and print accuracy
    accuracy = 100 * correct / total
    top3_accuracy = 100 * top3_correct / total
    print(f"Top-1 Accuracy on the test dataset: {accuracy:.2f}%") 
    print(f"Top-3 Accuracy on the test dataset: {top3_accuracy:.2f}%") 

    
    return 0



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the quantized ResNetN network.")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to the pre-trained weights file.")
    parser.add_argument("--use_coe", action="store_true", default=False, help="Use COE files for weights instead of a checkpoint.")
    args = parser.parse_args()

    output = test_quantized_network(args)