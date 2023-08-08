import torch
import torch.nn as nn
import torch.nn.functional as F
from .lpips import LPIPS
import random
from torchvision import models

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.multi_neg = args.multi_neg
        self.lpips = LPIPS(net='vgg', spatial=args.lpips_spatial, weight=args.layer_weight)

    def forward(self, sr, hr, lr):
        b, c, h, w = sr.shape
        b_, c_, h_, w_ = lr.shape
        if h_ != h or w_ != w:
            lr = F.interpolate(lr, (h, w), mode='bicubic', align_corners=True).clamp(0, 1)

        pos_lpips = self.lpips(sr, hr).mean()
        neg_lpips = self.lpips(sr, lr).mean() + 3e-7
        loss = pos_lpips / neg_lpips
        return loss


class RandContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.multi_neg = args.multi_neg
        self.lpips = LPIPS(net='vgg', spatial=args.lpips_spatial, weight=args.layer_weight)

    def forward(self, sr, hr_list, lr_list):
        b, c, h, w = sr.shape
        assert isinstance(lr_list, list)
        b_, c_, h_, w_ = lr_list[0].shape
        pos_lpips = 0
        if not isinstance(hr_list, list):
            hr_list = [hr_list, ]
        for hr in hr_list:
            pos_lpips = self.lpips(sr, hr).mean()
        pos_lpips /= len(hr_list)

        neg_lpips = 0
        for lr in lr_list:
            lr = lr.cuda() 
            if h_ != h or w_ != w:
                lr = F.interpolate(lr, (h, w), mode='bicubic', align_corners=True).clamp(0, 1)
            neg_lpips += self.lpips(sr, lr).mean()
        neg_lpips = neg_lpips / len(lr_list) + 3-7
        loss = pos_lpips / neg_lpips
        return loss


class MultiContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.multi_neg = args.multi_neg
        self.lpips = LPIPS(net='vgg', spatial=args.lpips_spatial, weight=args.layer_weight)
        self.neg = args.mcl_neg
        self.cl_loss_type = args.cl_loss_type

    def forward(self, sr, hr_list, lr_list):
        if not isinstance(lr_list, list):
            lr_list = [lr_list, ]
        if not isinstance(hr_list, list):
            hr_list = [hr_list]
        with torch.no_grad():
            pos_loss = self.cl_pos(sr, hr_list)
            neg_loss = self.cl_neg(sr, lr_list)
        loss = self.cl_loss(pos_loss, neg_loss)

        return loss

    def cl_pos(self, sr, hr_list):
        pos_loss = 0
        for hr in hr_list:
            pos_lpips = self.lpips(sr, hr).mean()
            pos_loss += pos_lpips
        pos_loss /= len(hr_list)
        return pos_loss

    def cl_neg(self, sr, lr_list):
        b, c, h, w = sr.shape
        batch_list = list(range(b))
        neg_lpips = 0
        for lr in lr_list:
            b_, c_, h_, w_ = lr.shape
            if h_ != h or w_ != w:
                lr = F.interpolate(lr, (h, w), mode='bicubic',  align_corners=True).clamp(0, 1)

            neg_lpips += self.lpips(sr, lr).mean()

            for neg_times in range(self.neg):
                random.shuffle(batch_list)
                neg_lpips_shuffle = self.lpips(sr, lr[batch_list, :, :, :]).mean()
                neg_lpips += neg_lpips_shuffle
        neg_lpips /= ((self.neg+1)*len(lr_list))
        return neg_lpips

    def cl_loss(self, pos_loss, neg_loss):

        if self.cl_loss_type in ['l2', 'cosine']:
            cl_loss = pos_loss - neg_loss

        elif self.cl_loss_type == 'l1':
            cl_loss = pos_loss / (neg_loss + 3e-7)
        else:
            raise TypeError(f'{self.args.cl_loss_type} not fount in cl_loss')

        return cl_loss


class LPIPSContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.multi_neg = args.multi_neg
        self.lpips = LPIPS(net='vgg', spatial=args.lpips_spatial, weight=args.layer_weight)

    def forward(self, sr, hr, lr):
        b, c, h, w = sr.shape
        b_, c_, h_, w_ = lr.shape
        if h_ != h or w_ != w:
            lr = F.interpolate(lr, (h, w), mode='bicubic',  align_corners=True).clamp(0, 1)

        pos_lpips = self.lpips(sr, hr).mean()
        neg_lpips = self.lpips(sr, lr).mean()
        loss = pos_lpips - 0.1*neg_lpips
        return loss


class LPIPSLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.multi_neg = args.multi_neg
        self.lpips = LPIPS(net='vgg', spatial=args.lpips_spatial, weight=args.layer_weight)

    def forward(self, sr, hr,):
        pos_lpips = self.lpips(sr, hr).mean()
        return pos_lpips


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGInfoNCE(nn.Module):
    def __init__(self, args):
        super(VGGInfoNCE, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.args = args
        self.cl_layer = [int(i.strip()) for i in args.cl_layer.split(',')]

    def infer(self, x):
        return self.vgg(x)

    def forward(self, sr, hr, lr):
        sr_features = self.vgg(sr)
        if not isinstance(hr, list):
            hr = [hr, ]
        if not isinstance(lr, list):
            lr = [lr, ]

        if self.args.pos_id != -1:
            hr = [hr[self.pos_id], ]

        loss = self.infoNCE(sr_features, hr, lr)
        return loss

    def infoNCE(self, sr_features, hr, lr):
        n_hr_features = []
        n_lr_features = []
        b, c, h, w = hr[0].shape

        with torch.no_grad():
            for s_hr in hr:
                n_hr_features.append(self.infer(s_hr))

            for s_lr in lr:
                b_, c_, h_, w_ = s_lr.shape
                if h_ != h or w_ != w:
                    s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, self.args.rgb_range)
                n_lr_features.append(self.infer(s_lr))

        infoNCE_loss = 0
        for l, idx in enumerate(self.cl_layer):
            sr_layer = sr_features[idx]
            hr_layers = []
            for hr_features in n_hr_features:
                hr_layers.append(hr_features[idx])

            lr_layers = []
            for lr_features in n_lr_features:
                lr_layers.append(lr_features[idx])
            if self.args.cl_loss_type == 'InfoNCE_L1':
                nce_loss = self.l1_nce(sr_layer, hr_layers, lr_layers)
            elif self.args.cl_loss_type == 'InfoNCE':
                nce_loss = self.nce(sr_layer, hr_layers, lr_layers)
            else:
                raise TypeError(f'{self.args.cl_loss_type} is not found')

            infoNCE_loss += nce_loss

        return infoNCE_loss / len(self.cl_layer)

    def l1_nce(self, sr_layer, hr_layers, lr_layers):

        loss = 0
        b, c, h, w = sr_layer.shape

        neg_logits = []

        for f_lr in lr_layers:
            neg_diff = torch.abs(sr_layer-f_lr).mean(dim=[-3, -2, -1]).unsqueeze(1)
            neg_logits.append(neg_diff)

        if self.args.shuffle_neg:
            batch_list = list(range(b))

            for f_lr in lr_layers:
                random.shuffle(batch_list)
                neg_diff = torch.abs(sr_layer-f_lr[batch_list, :, :, :]).mean(
                    dim=[-3, -2, -1]).unsqueeze(1)
                neg_logits.append(neg_diff)

        for f_hr in hr_layers:
            pos_logits = []
            pos_diff = torch.abs(sr_layer-f_hr).mean(dim=[-3, -2, -1]).unsqueeze(1)
            pos_logits.append(pos_diff)

            if self.args.cl_loss_type == 'InfoNCE':
                logits = torch.cat(pos_logits + neg_logits, dim=1)
                cl_loss = F.cross_entropy(logits, torch.zeros(b, device=logits.device, dtype=torch.long)) # self.ce_loss(logits)

            elif self.args.cl_loss_type == 'InfoNCE_L1':
                neg_logits = torch.cat(neg_logits, dim=1).mean(dim=1, keepdim=True)
                cl_loss = torch.mean(pos_logits[0] / neg_logits)

            elif self.args.cl_loss_type == 'LMCL':
                cl_loss = self.lmcl_loss(pos_logits + neg_logits)
            else:
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/cl.py')
            loss += cl_loss
        return loss / len(hr_layers)


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
                raise TypeError(f'{self.args.cl_loss_type} is not found in loss/cl.py')
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
