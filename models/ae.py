import torch
import torch.nn as nn
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module): # change the name of BinConv2d
    '''
    remove the batch normalization.
    '''
    def __init__(self, input_channels, output_channels, bn=True,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False, Transpose=False, previous_conv=False, size=0):
        super(BinConv2d, self).__init__()
        self.input_channels = input_channels
        # self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv
        self.non_linear = bn

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        self.Transpose = Transpose
        if not self.Linear and not self.Transpose:
            # self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        elif self.Transpose:
            self.transconv = nn.ConvTranspose2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            # self.linear = nn.Linear(input_channels, output_channels, bias=False)
            self.linear = nn.Linear(input_channels, output_channels) # should be check later. It is supposed to be non-biased. 
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # if self.non_linear:
        #     x = self.bn(x)
        x = BinActive.apply(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear and not self.Transpose:
            x = self.conv(x)
        elif self.Transpose:
            x = self.transconv(x)
        else:
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        if self.non_linear:
            x = self.bn(x)
        # x = self.relu(x)
        return x

class FCAutoEncoder(nn.Module):
    def __init__(self, dim_input, dim_latent):
        super(FCAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # BinConv2d(dim_input, dim_latent, bn=False, Linear=True))
            BinConv2d(dim_input, int(dim_input/2), bn=False, Linear=True),   # y1=Ax   x in [0,1], y1=
            # BinConv2d(dim_input, int(dim_input/4*3), bn=False, Linear=True),
            # BinConv2d(int(dim_input/4*3), int(dim_input/2), Linear=True),
            BinConv2d(int(dim_input/2), dim_latent, Linear=True))   # y2=1, if y1>p, else y2=0 ; y3 = A.y2, 

        self.decoder = nn.Sequential(
            # BinConv2d(dim_latent, dim_input, bn=False, Linear=True))
            BinConv2d(dim_latent, int(dim_input/2), bn=False, Linear=True),  # y1=Ax 
            # BinConv2d(dim_latent, int(dim_input/2), bn=False, Linear=True),
            # BinConv2d(int(dim_input/2), int(dim_input/4*3), Linear=True),
            BinConv2d(int(dim_input/2), dim_input, Linear=True)) # y2=1 y1>Pinteger

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class FCAutoEncoder1Layer(nn.Module):
    def __init__(self, dim_input, dim_latent):
        super(FCAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # BinConv2d(dim_input, dim_latent, bn=False, Linear=True))
            # BinConv2d(dim_input, int(dim_input/2), bn=False, Linear=True),   # y1=Ax   x in [0,1], y1=
            # # BinConv2d(dim_input, int(dim_input/4*3), bn=False, Linear=True),
            # # BinConv2d(int(dim_input/4*3), int(dim_input/2), Linear=True),
            BinConv2d(dim_input, dim_latent, Linear=True))   # y2=1, if y1>p, else y2=0 ; y3 = A.y2, 

        self.decoder = nn.Sequential(
            # BinConv2d(dim_latent, dim_input, bn=False, Linear=True))
            BinConv2d(dim_latent, dim_input, Linear=True),  # y1=Ax 
            # BinConv2d(dim_latent, int(dim_input/2), bn=False, Linear=True),
            # BinConv2d(int(dim_input/2), int(dim_input/4*3), Linear=True),
            # BinConv2d(int(dim_input/2), dim_input, Linear=True)) # y2=1 y1>Pinteger

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNNAutoEncoder(nn.Module):
    # def __init__(self):
    #     super(CNNAutoEncoder, self).__init__()
    #     self.encoder = nn.Sequential(
    #         BinConv2d(1, 8, 3, 2, 1, Linear=False, Transpose=True),   # 1*20*20 => 8*10*10
    #         BinConv2d(1, 16, 3, 2, 1, Linear=False, Transpose=True),  # 8*10*10 => 16*5*5

    #     )
    pass


class MLPClassifer(nn.Module):
    def __init__(self, input_size=784, ouput_size=10):
        super(MLPClassifer, self).__init__()
        self.classifer = nn.Sequential(
            BinConv2d(input_size, 500, bn=False, Linear=True),
            BinConv2d(500, 100, Linear=True),
            BinConv2d(100, 10, bn=False, Linear=True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.classifer(x)
        return output