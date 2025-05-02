import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
import custom_spikingjelly_class
from utilities import Q3_5QuantizedTensor, Q3_5Hook

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
        # Perform preprocessing outside of the model
        #if(not self.args.use_coe):
        #    with torch.no_grad():
        #        x = 3*x/(torch.max(x))
        #if(not self.args.use_coe):
        #    x = self.maxpool0(x)

        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]

        x = self.conv2d_1(x)

        if(not self.args.use_coe):
            x = self.bn1(x) 
        x = self.LIF1(x) 
        x = self.maxpool1(x) 

        x = self.flat(x)
        x = self.out(x.mean(0))

        return x