import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'WaveL1':
                cl = import_module('loss.loss')
                loss_function = cl.WaveL1Loss(args)
            elif loss_type == 'MultiL1':
                loss_function = nn.L1Loss()
            elif loss_type == 'MultiWaveL1':
                cl = import_module('loss.loss')
                loss_function = cl.MultiWaveL1Loss(args)
            elif loss_type == 'CL':
                cl = import_module('loss.cl')
                loss_function = cl.ContrastiveLoss(
                    args
                )
            elif loss_type == 'RCL':
                assert args.random_neg
                cl = import_module('loss.cl')
                loss_function = cl.RandContrastiveLoss(
                    args
                )
            elif loss_type == 'MCL':
                cl = import_module('loss.cl')
                loss_function = cl.MultiContrastiveLoss(
                    args
                )
            elif loss_type == 'LCL':
                cl = import_module('loss.cl')
                loss_function = cl.LPIPSContrastiveLoss(
                    args
                )
            elif loss_type == 'VGGInfoNCE':
                cl = import_module('loss.cl')
                loss_function = cl.VGGInfoNCE(
                    args
                )
            elif loss_type == 'LPIPS':
                cl = import_module('loss.cl')
                loss_function = cl.LPIPSLoss(
                    net='vgg',
                    spatial=args.lpips_spatial
                )
            elif loss_type.find('WaveD') >= 0:
                module = import_module('loss.discriminator')
                loss_function = getattr(module, 'WaveDLoss')(
                    args
                )
            elif loss_type.find('ContrasD') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type.find('patchD') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type.find('CLD') >= 0:
                module = import_module('loss.discriminator')
                loss_function = getattr(module, 'CLDLoss')(
                    args
                )
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.9f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module,
            )
        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr, lr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] in ['CL', 'MCL', 'LCL', 'CL-GAN', 'CL-RGAN', 'RCL', 'MCL-RGAN', 'CL-GAN',
                                 'MCL-GAN', 'WaveD', 'CLD', 'CL-ContrasD', 'ContrasD',
                                 'LSGAN', 'CL-LSGAN', 'VGGInfoNCE', 'CL-patchD', 'patchD']:
                    loss = l['function'](sr, hr, lr)
                else:
                    loss = l['function'](sr, hr[0])
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.7f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath, epoch):
        torch.save(self.state_dict(), os.path.join(apath, f'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

