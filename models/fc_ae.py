import torch
import torch.nn as nn
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    # @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    # @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module): # change the name of BinConv2d
    '''
    remove the batch normalization.
    '''
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            # self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            # if self.previous_conv:
            #     self.bn = nn.BatchNorm2d(int(input_channels/size), eps=1e-4, momentum=0.1, affine=True)
            # else:
            #     self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        # self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x = self.bn(x)
        x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        # x = self.relu(x)
        return x

class FCAutoEncoder(nn.Module):
    def __init__(self):
        super(FCAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            BinConv2d(1000, 500, Linear=True),
            BinConv2d(500, 250, Linear=True),
            BinConv2d(250, 100, Linear=True))

        self.decoder = nn.Sequential(
            BinConv2d(100, 250, Linear=True),
            BinConv2d(250, 500, Linear=True),
            # nn.Linear(500, 1000))
            BinConv2d(250, 1000, Linear=True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # x = BinActive.apply(x)
        return x