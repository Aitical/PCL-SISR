from model import common
from pytorch_wavelets import DWTForward
import torch.nn as nn
import torch
import random
import torch.nn.functional as F
from collections import OrderedDict

class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        in_channels = 9 if args.use_wavelet else args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
            )

        m_features = [_block(in_channels, out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))
            m_features.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2**((depth + 1) // 2))
        if args.use_wavelet:
            patch_size = patch_size // 2

        if not 'patchD' in args.loss:

            m_classifier = [
                nn.Flatten(),
                nn.Linear(out_channels * patch_size**2, 1024),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(1024, 1)
            ]
            self.classifier = nn.Sequential(*m_classifier)
        else:
            m_classifier = [
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(out_channels, 1, kernel_size=(1, 1))
            ]
            self.classifier = nn.Sequential(*m_classifier)

        if args.use_wavelet:
            self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
            self.filter = self.filter_wavelet
            self.cs = 'cat'

    def forward(self, x, return_features=False):
        if self.args.use_wavelet:
            x = self.filter(x)

        if return_features:
            all_feature = []
            x_in = x
            for layer in self.features:
                if self.args.vgg_like_relu:
                    if isinstance(layer, nn.LeakyReLU):
                        continue
                x_in = layer(x_in)
                if self.args.before_relu and not isinstance(layer, nn.LeakyReLU):
                    all_feature.append(x_in)
                elif not self.args.before_relu and isinstance(layer, nn.LeakyReLU):
                    all_feature.append(x_in)

            output = self.classifier(x_in)
            return output, all_feature
        else:
            features = self.features(x)
            output = self.classifier(features)

            return output

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


class WaveDLoss(nn.Module):
    '''
        output is not normalized
    '''

    def __init__(self, args):
        super(WaveDLoss, self).__init__()
        self.args = args
        in_channels = 9
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
            )

        m_features = [_block(in_channels, out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))
            m_features.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))

        if args.use_wavelet:
            patch_size = patch_size // 2

        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

        if args.use_wavelet:
            self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
            self.filter = self.filter_wavelet
            self.cs = 'cat'

        self.load()
        for p in self.parameters():
            p.requires_grad = False

    def infer(self, x):

        x = self.filter(x)
        all_feature = []
        x_in = x
        for layer in self.features:
            x_in = layer(x_in)
            if not isinstance(layer, nn.LeakyReLU):
                all_feature.append(x_in)

        return all_feature

    def forward(self, sr, hr, lr):
        sr_features = self.infer(sr)
        if not isinstance(hr, list):
            hr = [hr, ]
        if not isinstance(lr, list):
            lr = [lr, ]

        pos_loss = 0
        for s_hr in hr:
            s_hr_features = self.infer(s_hr)
            pos_diff = self.cl_pos(sr_features, s_hr_features)
            pos_loss += pos_diff
        pos_loss /= len(hr)

        neg_loss = 0
        for s_lr in lr:
            s_lr_features = self.infer(s_lr)
            neg_diff = self.cl_neg(sr_features, s_lr_features)
            neg_loss += neg_diff
        neg_loss /= len(lr)
        cl_loss = neg_loss - pos_loss
        return cl_loss

    def cl_pos(self, sr_layers, hr_layers):
        """
        sr_layers: feature map list
        hr_layers: feature map list
        """
        pos_loss = 0
        for l, idx in enumerate(self.cl_layer):
            pos_diff = torch.sum(
                F.normalize(sr_layers[idx], dim=1) * F.normalize(hr_layers[idx], dim=1), dim=1).mean()
            if self.use_weights:
                pos_diff /= 2 ** l
            pos_loss += pos_diff
        return pos_loss

    def cl_neg(self, sr_layers, lr_layers):
        neg_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_feature = F.normalize(sr_layers[idx], dim=1)
            lr_feature = F.normalize(lr_layers[idx], dim=1)
            neg_diff = torch.sum(
                 sr_feature * lr_feature, dim=1).mean()

            if self.args.multi_neg_D:
                batch_list = list(range(lr_layers[idx].shape[0]))
                for _ in range(self.args.mcl_neg):
                    random.shuffle(batch_list)
                    neg_shuffle = torch.sum(sr_feature * lr_feature[batch_list, :, :, :], dim=1).mean()
                    neg_diff += neg_shuffle
                neg_diff /= (self.args.mcl_neg+1)

            if self.use_weights:
                neg_diff /= 2**l
            neg_loss += neg_diff
        return neg_loss

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

    def load(self,):
        loss_dict = torch.load(self.args.WaveD_path, map_location='cpu')
        dis_dict = OrderedDict()
        for k, v in loss_dict.items():
            if k.startswith('loss_module.1.'):
                dis_dict[k.replace('loss_module.1.', '')] = v
        miss = self.load_state_dict(dis_dict)
        print(f'Load from {self.args.WaveD_path}', miss)


class CLDLoss(nn.Module):
    '''
        output is not normalized
    '''

    def __init__(self, args):
        super(CLDLoss, self).__init__()
        self.args = args
        in_channels = 3
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
            )
        self.pos_id = int(self.args.pos_id)
        m_features = [_block(in_channels, out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))
            m_features.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))

        #m_classifier = [
        #    nn.Linear(out_channels * patch_size ** 2, 1024),
        #    nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #    nn.Linear(1024, 1)
        #]

        self.features = nn.Sequential(*m_features)
        #self.classifier = nn.Sequential(*m_classifier)

        self.load()
        for p in self.parameters():
            p.requires_grad = False
        self.cl_layer = [int(i.strip()) for i in args.cl_layer.split(',')]
        self.args = args
        self.use_weights = args.layer_weight

    def infer(self, x):

        all_feature = []
        x_in = x
        for layer in self.features:
            x_in = layer(x_in)
            if not isinstance(layer, nn.LeakyReLU):
                all_feature.append(x_in)

        return all_feature

    def forward(self, sr, hr, lr):
        # sr_features = self.infer(sr)
        if not isinstance(hr, list):
            hr = [hr, ]
        if not isinstance(lr, list):
            lr = [lr, ]

        if self.pos_id != -1:
            hr = [hr[self.pos_id], ]

        if not self.args.cl_loss_type in ['InfoNCE', 'LMCL']:
            loss = self.cl_loss(sr, hr, lr)
        else:
            loss = self.cl_infoNCE(sr, hr, lr)
        return loss

    def cl_loss(self, sr, hr, lr):
        sr_features = self.infer(sr)

        pos_loss = 0
        for s_hr in hr:
            s_hr_features = self.infer(s_hr)
            pos_diff = self.cl_pos(sr_features, s_hr_features)
            pos_loss += pos_diff
        pos_loss /= len(hr)

        neg_loss = 0
        b, c, h, w = sr.shape
        for s_lr in lr:
            b_, c_, h_, w_ = s_lr.shape
            if h_ != h or w_ != w:
                s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, 1)
            s_lr_features = self.infer(s_lr)
            neg_diff = self.cl_neg(sr_features, s_lr_features)
            neg_loss += neg_diff
        neg_loss /= len(lr)
        cl_loss = neg_loss - pos_loss
        return cl_loss

    def cl_infoNCE(self, sr, hr, lr):
        sr_features = self.infer(sr)
        n_hr_features = []
        n_lr_features = []
        for s_hr in hr:
            n_hr_features.append(self.infer(s_hr))
        b, c, h, w = s_hr.shape
        for s_lr in lr:
            b_, c_, h_, w_ = s_lr.shape
            if h_ != h or w_ != w:
                s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, 1)
            n_lr_features.append(self.infer(s_lr))

        nce_loss = 0
        for l, idx in enumerate(self.cl_layer):
            lr_layers = []
            for n_lr in n_lr_features:
                lr_layers.append(n_lr[idx])
            # TODO: infoNCE uses one postive only?
            hr_layers = []
            for n_hr in n_hr_features:
                hr_layers.append(n_hr[idx])

            loss = self.cl_exp(sr_features[idx], hr_layers, lr_layers)
            nce_loss += loss

        return nce_loss / len(self.cl_layer)

    def cl_exp(self, sr_layer, hr_layers, lr_layers):

        loss = 0
        b, c, h, w = sr_layer.shape

        neg_logits = []

        for f_lr in lr_layers:
            neg_diff = torch.sum(
                F.normalize(sr_layer, dim=1) * F.normalize(f_lr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            neg_logits.append(neg_diff)

        if self.args.shuffle_neg:
            batch_list = list(range(b))

            for f_lr in lr_layers:
                random.shuffle(batch_list)
                neg_diff = torch.sum(
                    F.normalize(sr_layer, dim=1) * F.normalize(f_lr[batch_list, :, :, :], dim=1), dim=1).mean(
                    dim=[-1, -2]).unsqueeze(1)
                neg_logits.append(neg_diff)

        for f_hr in hr_layers:
            pos_logits = []
            pos_diff = torch.sum(
                F.normalize(sr_layer, dim=1) * F.normalize(f_hr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            pos_logits.append(pos_diff)

            if self.args.cl_loss_type == 'InfoNCE':
                logits = torch.cat(pos_logits + neg_logits, dim=1)
                cl_loss = F.cross_entropy(logits, torch.zeros(b, device=logits.device, dtype=torch.long)) # self.ce_loss(logits)
            elif self.args.cl_loss_type == 'LMCL':
                cl_loss = self.lmcl_loss(pos_logits + neg_logits)
            else:
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/discriminator.py')
            loss += cl_loss
        return loss / len(hr_layers)

    def lmcl_loss(self, logits):
        """
        logits: BXK, the first column is the positive similarity
        """
        pos_sim = logits[0]
        neg_sim = torch.cat(logits[1:], dim=1)
        pos_logits = pos_sim.exp()  # Bx1
        neg_logits = torch.sum(neg_sim.exp(), dim=1, keepdim=True)  # Bx1
        loss = -torch.log(pos_logits / neg_logits).mean()
        return loss

    def cl_pos(self, sr_layers, hr_layers):
        """
        sr_layers: feature map list
        hr_layers: feature map list
        """
        pos_loss = 0
        for l, idx in enumerate(self.cl_layer):
            pos_diff = torch.sum(
                F.normalize(sr_layers[idx], dim=1) * F.normalize(hr_layers[idx], dim=1), dim=1).mean()
            if self.use_weights:
                pos_diff /= 2 ** l
            pos_loss += pos_diff
        return pos_loss

    def cl_neg(self, sr_layers, lr_layers):
        neg_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_feature = F.normalize(sr_layers[idx], dim=1)
            lr_feature = F.normalize(lr_layers[idx], dim=1)
            neg_diff = torch.sum(
                 sr_feature * lr_feature, dim=1).mean()

            if self.args.shuffle_neg:
                batch_list = list(range(lr_layers[idx].shape[0]))
                for _ in range(self.args.mcl_neg):
                    random.shuffle(batch_list)
                    neg_shuffle = torch.sum(sr_feature * lr_feature[batch_list, :, :, :], dim=1).mean()
                    neg_diff += neg_shuffle
                neg_diff /= (self.args.mcl_neg+1)

            if self.use_weights:
                neg_diff /= 2**l
            neg_loss += neg_diff
        return neg_loss


    def load(self,):
        loss_dict = torch.load(self.args.CLD_path, map_location='cpu')
        dis_dict = OrderedDict()
        for k, v in loss_dict.items():
            if k.find('classifier')>=0:
               continue
            if k.startswith('loss_module.1.'):
                dis_dict[k.replace('loss_module.1.', '')] = v
           
        miss = self.load_state_dict(dis_dict)
        print(f'Load from {self.args.CLD_path}', miss)
