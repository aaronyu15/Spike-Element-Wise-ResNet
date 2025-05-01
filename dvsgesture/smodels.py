import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import layer

class ResNetN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()

        self.maxpool0 = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2d_1 = layer.SeqToANNContainer(nn.Conv2d(2, 6, kernel_size=3, padding=0, stride=1, bias=False))
        self.bn1 = layer.SeqToANNContainer(nn.BatchNorm2d(6))
        self.LIF1 = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
        self.maxpool1 = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2))

        self.flat = nn.Flatten(2)

        with torch.no_grad():
            x = torch.zeros([1, 2, 128, 128])
            for m in self.modules():
                if isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Conv2d):
                    x = m(x)
            out_features = x.numel() 

        self.out = nn.Linear(out_features, num_classes, bias=False)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = 3*x/(torch.max(x))
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]

        x = self.maxpool0(x)

        x = self.conv2d_1(x)
        x = self.bn1(x) 
        x = self.LIF1(x) 
        x = self.maxpool1(x) 

        x = self.flat(x)
        x = self.out(x.mean(0))

        return x

def SEWResNet():
    num_classes = 11
    return ResNetN(num_classes)