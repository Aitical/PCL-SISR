import utility
from types import SimpleNamespace

from model import common
from loss import discriminator
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.dis = discriminator.Discriminator(args)
        if gan_type == 'WGAN_GP':
            # see https://arxiv.org/pdf/1704.00028.pdf pp.4
            optim_dict = {
                'optimizer': 'ADAM',
                'betas': (0, 0.9),
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': args.weight_decay,
                'decay': args.decay,
                'gamma': args.gamma
            }
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args
        self.use_weights = args.layer_weight
        self.cl_layer = [int(l.strip()) for l in args.cl_layer.split(',')]
        self.optimizer = utility.make_optimizer(optim_args, self.dis)
        self.l1 = nn.L1Loss(reduction='mean')
        self.args = args

    def forward(self, fake, real_sample, neg_sample=None):
        # updating discriminator...
        self.loss = 0
        if isinstance(real_sample, list):
            real = real_sample[0]
        else:
            real = real_sample
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            if self.args.contras_D_train:
                real_aug = []
                fake_aug = []
                for r_ in real_sample[1:]:
                    real_aug.append(self.dis(r_))
                for n_ in neg_sample[1:]:
                    fake_aug.append(self.dis(n_))
                fake_aug.append(d_fake)
                real_aug.append(d_real)
                d_fake = torch.cat(fake_aug, dim=0)
                d_real = torch.cat(real_aug, dim=0)
            retain_graph = False
            if self.gan_type in ['GAN', 'CL-GAN']:
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            elif self.gan_type.find('LSGAN') >= 0:
                real_label = torch.zeros_like(d_real, device=d_real.device)+1.0

                loss_d = F.mse_loss(d_real, real_label) + F.mse_loss(d_fake, -1*real_label)
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif self.gan_type in ['RGAN', 'CL-RGAN']:
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True
            elif self.gan_type.find('ContrasD') >= 0:
                loss_d = self.contrastive_D_loss(d_real, d_fake)
            elif 'patchD' in self.gan_type:
                # patch logits reshape using dual contrastive loss
                loss_d = self.contrastive_D_loss(d_real.reshape(-1), d_fake.reshape(-1))

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        # updating generator...
        with torch.no_grad():
            d_fake_bp, fake_features = self.dis(fake, return_features=True)      # for backpropagation, use fake as it is
            d_real_bp, real_features = self.dis(real, return_features=True)

        if self.args.no_ad_loss:
            loss_g = 0.0
        elif self.gan_type in ['GAN', 'CL-GAN']:
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type.find('LSGAN') >= 0:
            loss_g = torch.mean(d_fake_bp**2)
        elif self.gan_type in ['RGAN', 'CL-RGAN']:
            better_real = d_real_bp - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real_bp.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)
        elif self.gan_type.find('ContrasD') >= 0:
            loss_g = self.contrastive_D_loss(d_fake_bp, d_real_bp)
        elif 'patchD' in self.gan_type:
            loss_g = self.contrastive_D_loss(d_fake_bp.reshape(-1), d_real_bp.reshape(-1))
        else:
            raise TypeError(f'{self.gan_type} is not found')

        if 'CL' in self.gan_type:
            if not self.args.cl_loss_type in ['InfoNCE', 'LMCL']:
                cl_loss = self.cl_l1_loss(fake_features, real_sample, neg_sample)
            else:
                cl_loss = self.infoNCE(fake_features, real_sample, neg_sample)

            loss_g += self.args.cl_gan_cl_weight * cl_loss
        return loss_g

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
                    s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, self.args.rgb_range)
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
                    s_lr = F.interpolate(s_lr, size=(h, w), align_corners=True, mode='bicubic').clamp(0, self.args.rgb_range)
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

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss
               
