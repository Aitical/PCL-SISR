from turtle import forward
from matplotlib.pyplot import isinteractive
from numpy import isin
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class WaveL1Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.use_wavelet:
            self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
            self.filter = self.filter_wavelet
            self.cs = 'cat'
            assert not args.use_fft

        elif args.use_fft:
            self.filter = torch.fft.rfft2

        self.l1 = nn.L1Loss()

    def filter_wavelet(self, x, norm=True):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs.lower() == 'sum':
            return (LH + HL + HH) / 3.
        elif self.cs.lower() == 'cat':
            return torch.cat((LH, HL, HH), 1)
        else:
            raise NotImplementedError('Wavelet format [{:s}] not recognized'.format(self.cs))

    def forward(self, x, y):
        x_high = self.filter(x)
        y_high = self.filter(y)
        loss = self.l1(x_high, y_high)
        return loss
    
class MultiWaveL1Loss(WaveL1Loss):
    def __init__(self, args):
        super().__init__(args)
    
    def forward(self, x, y):
        if not isinstance(y, list):
            y = [y, ]
        loss = 0
        for t in y:
            loss += super().forward(x, t)
        loss /= (len(y)*1.0)

        return loss

class MultiL1(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        if not isinstance(y, list):
            y = [y, ]
        loss = 0
        for t in y:
            loss += self.l1(x, t)
        loss /= (len(y)*1.0)

        return loss
        