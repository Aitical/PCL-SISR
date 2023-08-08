import os
import math
import random
from decimal import Decimal
from data.div2k import RandomDegrad
import torchvision.transforms.functional as F
import torchvision
import utility
from kornia.enhance import sharpness
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from PIL import Image
import numpy as np


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        if args.use_wandb:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            if args.wandb_offline:
                os.environ["WANDB_MODE"] = "offline"
            else:
                os.environ["WANDB_MODE"] = "online"

            self.run_wb = wandb.init(config=vars(args), project=args.proj_name,
                                     name=args.exp_name, notes=args.loss,
                                     dir=args.wandb_dir,
                                     tags=[f'{args.scale}'])
        else:
            self.run_wb = None

        if isinstance(my_model, list):
            self.model = my_model[0]
            self.ema_model = my_model[1]
            assert args.use_ema
        else:
            self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.random_degrad = RandomDegrad(args)
        print(vars(args))

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        if self.run_wb:
            self.run_wb.log({'lr': lr, 'epoch': epoch})
        self.ckp.write_log(
            '{} [Epoch {}]\tLearning rate: {:.2e}'.format(self.args.exp_name, epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model, timer_aug, timer_loss = utility.timer(), utility.timer(), utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr_cpu, hr_cpu, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr_cpu, hr_cpu)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            timer_model.hold()
            timer_aug.tic()
                    
            if self.args.use_aug:
                random_lr_neg = [lr,]
                if self.args.random_neg:
                      # REVIEW: add bicubic neg sample
                    for i in range(self.args.mcl_neg):
                        r_neg = self.random_degrad(hr) if self.args.gpu_blur else self.random_degrad(hr_cpu).cuda()
                        if self.args.neg_sr:
                            assert self.args.random_neg and not self.args.only_blur
                            with torch.no_grad():
                                r_neg = self.model(r_neg, 0)
                        random_lr_neg.append(r_neg)
                
                random_sharp_hr = [hr, ]
                if self.args.sharp_hr:
                    if self.args.gpu_sharp:
                        for i in range(self.args.mcl_neg):
                            # Batch value
                            sharp_value = torch.rand(hr.shape[0])*self.args.sharp_range+self.args.sharp_value
                            ## TODO: review sharpen in augmentation
                            sharpened_hr = sharpness(hr/(self.args.rgb_range*1.0), sharp_value)*self.args.rgb_range
                            random_sharp_hr.append(utility.quantize(sharpened_hr, self.args.rgb_range))
                    else:
                        for i in range(self.args.mcl_neg):
                            sharp_value = random.randint(25, 60)/10
                            s_hr = F.adjust_sharpness(hr, sharp_value)
                            random_sharp_hr.append(s_hr)

                if self.args.use_noise_pos:
                    noise_scale = self.args.noise_pos_value
                    for i in range(self.args.mcl_neg):
                        pos_noise = torch.randn_like(hr) * noise_scale
                        random_sharp_hr.append(hr+pos_noise)
                timer_aug.hold()
                timer_loss.tic()
                loss = self.loss(sr, random_sharp_hr, random_lr_neg)
                timer_loss.hold()
            else:
                timer_loss.tic()
                loss = self.loss(sr, [hr,], [lr,])
                timer_loss.hold()

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            if self.run_wb:
                self.run_wb.log({'train_loss': loss.item()})

            if (batch + 1) % self.args.print_every == 0:
                time_model_c = timer_model.release()
                time_data_c = timer_data.release()
                time_aug_c = timer_aug.release()
                time_loss_c = timer_loss.release()
                for l_, v_ in zip(self.loss.loss, self.loss.log[-1]):
                    if self.run_wb:
                        self.run_wb.log({l_['type']+' Loss': v_/(batch+1), 'epoch': epoch})

                self.ckp.write_log('[{}/{}]\t{}\t model: {:.1f}+ data: {:.1f}s+ aug: {:.1f}s+ loss: {:.1f}s = {:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    time_model_c,
                    time_data_c,
                    time_aug_c,
                    time_loss_c,
                    sum([time_model_c, time_data_c, time_loss_c, time_aug_c])
                ))

            if (batch + 1) % self.args.ema_iter == 0 and self.args.use_ema:
                for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
                    param_k.data.copy_(0.99*param_k.data + 0.01*param_q.data)  # initialize
                    param_k.requires_grad = False  # not update by gradient

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.ckp.add_ssim(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    ssim_value = utility.calc_ssim(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log_ssim[-1, idx_data, idx_scale] += ssim_value
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale, epoch)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.log_ssim[-1, idx_data, idx_scale] /= len(d)
                
                best = self.ckp.log.max(0)
                if self.run_wb:
                    self.run_wb.log({f'Test {d.dataset.name} PSNR': self.ckp.log[-1, idx_data, idx_scale], f'Test {d.dataset.name} SSIM': self.ckp.log_ssim[-1, idx_data, idx_scale], 'epoch': epoch},)
                self.ckp.write_log(
                    '{} [{} x{}]\tPSNR: {:.5f} \t SSIM {:.5f} (Best: {:.5f} @epoch {})'.format(
                        self.args.exp_name,
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        self.ckp.log_ssim[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

