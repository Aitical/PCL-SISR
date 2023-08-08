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
            assert not args.use_fft
        elif args.use_fft:
            patch_size = patch_size

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
            assert not args.use_fft
        elif args.use_fft:
            self.filter = torch.fft.rfft2

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

    def filter_wavelet(self, x, norm=False):
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
            nn.Flatten(),
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

        self.features = nn.Sequential(*m_features)

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
        cl_loss = self.infoNCE(sr, hr, lr)
        return cl_loss

    def infoNCE(self, sr_features, hr, lr):
        n_hr_features = []
        n_lr_features = []
        b, c, h, w = hr[0].shape

        if self.args.pos_id != -1:
            hr = [hr[self.args.pos_id], ]
        if self.args.neg_id != -1:
            lr = [lr[self.args.neg_id], ]

        if self.args.only_aug:
            hr = hr[1:]
            lr = lr[1:]

        with torch.no_grad():
            for s_hr in hr:
                n_hr_features.append(self.dis(s_hr, return_features=True)[1])

            for s_lr in lr:
                b_, c_, h_, w_ = s_lr.shape
                if h_ != h or w_ != w:
                    s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, 1)
                n_lr_features.append(self.dis(s_lr, return_features=True)[1])

        infoNCE_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_layer = sr_features[idx]
            hr_layers = []
            for hr_features in n_hr_features:
                hr_layers.append(hr_features[idx])

            lr_layers = []
            for lr_features in n_lr_features:
                lr_layers.append(lr_features[idx])
            nce_loss = self.nce(sr_layer, hr_layers, lr_layers)
            infoNCE_loss += nce_loss

        return infoNCE_loss / len(self.cl_layer)

    def nce(self, sr_layer, hr_layers, lr_layers):

        loss = 0
        b, c, h, w = sr_layer.shape

        neg_logits = []

        for f_lr in lr_layers:
            neg_diff = torch.sum(
               F.normalize(sr_layer, dim=1) * F.normalize(f_lr, dim=1), dim=1).mean(dim=[-1, -2]).unsqueeze(1)
            neg_logits.append(neg_diff)

        if self.args.shuffle_neg:
            batch_list = list(range(b))
            for t_ in range(self.args.shuffle_neg_num):
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
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/adversarial.py')
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


    def cl_l1_loss(self, fake_features, real_sample, neg_sample):
        b, c, h, w = real_sample[0].shape
        pos_loss = 0
        if not isinstance(real_sample, list):
            real_sample = [real_sample]
        for s_real in real_sample:
            with torch.no_grad():
                _, s_real_features = self.dis(s_real, return_features=True)
            pos_diff = self.cl_pos(fake_features, s_real_features)
            pos_loss += pos_diff
        pos_loss /= len(real_sample)

        if not isinstance(neg_sample, list):
            neg_sample = [neg_sample]
        neg_loss = 0
        for s_lr in neg_sample:
            with torch.no_grad():
                b_, c_, h_, w_ = s_lr.shape
                if h_ != h or w_ != w:
                    s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, 1)
                _, s_lr_features = self.dis(s_lr, return_features=True)
            neg_diff = self.cl_neg(fake_features, s_lr_features)
            neg_loss += neg_diff
        neg_loss /= len(neg_sample)
        cl_loss = self.cl_loss(pos_loss, neg_loss)
        return cl_loss

    def contrastive_D_loss(self, real_logits, fake_logits):
        device = real_logits.device
        real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

        def loss_half(t1, t2):
            t1 = rearrange(t1, 'i -> i ()')
            t2 = repeat(t2, 'j -> i j', i=t1.shape[0])
            t = torch.cat((t1, t2), dim=-1)
            return F.cross_entropy(t, torch.zeros(t1.shape[0], device=device, dtype=torch.long))

        return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)
    def cl_pos(self, sr_layers, hr_layers):
        if self.args.cl_loss_type == 'l1':
            return self.cl_pos_l1(sr_layers, hr_layers)
        elif self.args.cl_loss_type in ['l2', 'cosine']:
            return self.cl_pos_l2(sr_layers, hr_layers)
        else:
            raise TypeError(f'{self.args.cl_loss_type} was not found in cl_pos')

    def cl_neg(self, sr_layers, lr_layers):
        if self.args.cl_loss_type == 'l1':
            return self.cl_pos_l1(sr_layers, lr_layers)
        elif self.args.cl_loss_type in ['l2', 'cosine']:
            return self.cl_pos_l2(sr_layers, lr_layers)
        else:
            raise TypeError(f'{self.args.cl_loss_type} was not found in cl_neg')

    def cl_pos_l2(self, sr_layers, hr_layers):
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

    def cl_pos_l1(self, sr_layers, hr_layers):
        """
        sr_layers: feature map list
        hr_layers: feature map list
        """
        pos_loss = 0
        for l, idx in enumerate(self.cl_layer):
            pos_diff = self.l1(sr_layers[idx], hr_layers[idx])
            if self.use_weights:
                pos_diff /= 2 ** l
            pos_loss += pos_diff
        return pos_loss

    def cl_neg_l2(self, sr_layers, lr_layers):
        neg_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_feature = F.normalize(sr_layers[idx], dim=1)
            lr_feature = F.normalize(lr_layers[idx], dim=1)
            neg_diff = torch.sum(
                 sr_feature * lr_feature, dim=1).mean()
            if 'MCL' in self.gan_type:
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

    def cl_neg_l1(self, sr_layers, lr_layers):
        neg_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_feature = sr_layers[idx]
            lr_feature = lr_layers[idx]
            neg_diff = self.l1(sr_feature, lr_feature)

            if 'MCL' in self.gan_type:
                batch_list = list(range(lr_layers[idx].shape[0]))
                for _ in range(self.args.mcl_neg):
                    random.shuffle(batch_list)
                    neg_shuffle = self.l1(sr_feature * lr_feature[batch_list, :, :, :])
                    neg_diff += neg_shuffle
                neg_diff /= (self.args.mcl_neg+1)

            if self.use_weights:
                neg_diff /= 2**l

            neg_loss += neg_diff
        return neg_loss

    def cl_loss(self, pos_loss, neg_loss):
        if self.args.cl_loss_type == 'cosine':
            cl_loss = neg_loss - pos_loss
        elif self.args.cl_loss_type == 'l2':
            cl_loss = (1-pos_loss) / (1-neg_loss + 3e-7)
        elif self.args.cl_loss_type == 'l1':
            cl_loss = pos_loss / (neg_loss + 3e-7)
        else:
            raise TypeError(f'{self.args.cl_loss_type} not fount in cl_loss')

        return cl_loss

    def filter_wavelet(self, x, norm=True):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
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
