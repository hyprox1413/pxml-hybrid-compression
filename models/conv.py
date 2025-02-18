import torch

from torch import nn

class ConvolutionalEncoder(torch.nn.Module):
    def __init__(self, num_filters, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size, stride=stride)

    def forward(self, x):
        x = self.conv1(x)
        return x

class DeconvolutionalDecoder(torch.nn.Module):
    def __init__(self, num_filters, kernel_size, stride):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(num_filters, 3, kernel_size, stride=stride)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.deconv1(x)
        # x = (self.tanh(x) + 1) * 128
        return x

class Autoencoder(torch.nn.Module):
    def __init__(self, num_filters=8, kernel_size=10, stride=10):
        super().__init__()
        self.encoder = ConvolutionalEncoder(num_filters, kernel_size, stride)
        self.decoder = DeconvolutionalDecoder(num_filters, kernel_size, stride)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
