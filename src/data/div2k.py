import os
from data import srdata, common
from scipy.io import loadmat
from PIL import Image
import torch
import numpy as np
import cv2
import os.path
import random
import torchvision.transforms as transform
from kornia.filters import gaussian_blur2d
from utility import quantize

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        if self.input_large: self.dir_lr += 'L'

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pil_hr = Image.fromarray(hr)

        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

class RandomDegrad():
    def __init__(self, args):
        super(RandomDegrad, self).__init__()
        self.transform = transform.ToTensor()
        self.blur = args.only_blur
        self.args = args

    def AddNoise(self, img):  # noise
        # if random.random() > 0.9: #
        #     return img
        self.sigma = np.random.randint(1, 15)
        img_tensor = torch.from_numpy(np.array(img)).float()
        noise = torch.randn(img_tensor.size()).mul_(self.sigma / 1.0)

        noiseimg = torch.clamp(noise + img_tensor, 0, 255)
        return Image.fromarray(np.uint8(noiseimg.numpy()))

    def AddBlur(self, img):  # gaussian blur or motion blur
        # if random.random() > 0.9: #
        #     return img
        # img = np.array(img)
        if random.random() > 0.35:  ##gaussian blur
            blursize = random.randint(1, 5) * 2 + 1  ##3,5,7,9,11,13,15
            blursigma = random.randint(2, 10)
            img = cv2.GaussianBlur(img, (blursize, blursize), blursigma * 1.0)
        else:  # motion blur
            M = random.randint(1, 32)
            KName = './data/MotionBlurKernel/m_%02d.mat' % M
            k = loadmat(KName)['kernel']
            k = k.astype(np.float32)
            k /= np.sum(k)
            img = cv2.filter2D(img, -1, k)
        return Image.fromarray(img)

    def AddDownSample(self, img, size):  # downsampling
        # if random.random() > 0.95: #
        #     return img
        sampler = random.randint(4, 16) * 1.0
        img = img.resize((int(size / sampler), int(size / sampler)), Image.BICUBIC)
        return img

    def AddJPEG(self, img):  # JPEG compression
        # if random.random() > 0.6: #
        #     return img
        imQ = random.randint(70, 95)
        img = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), imQ]  # (0,100),higher is better,default is 95
        _, encA = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encA, 1)
        return Image.fromarray(img)

    def AddUpSample(self, img, size):
        return img.resize((size, size), Image.BICUBIC)

    def __call__(self, hr):
        b, c, h, w = hr.shape
        size = h//self.args.scale[0]

        if self.args.gpu_blur:
            kx = random.randint(1, 5) * 2 + 1
            ky = random.randint(1, 5) * 2 + 1
            sx = random.random() * 1.9 + 0.1
            sy = random.random() * 1.9 + 0.1
            # TODO: REVIEW rgb range! Operations in Kornia are taken normalized image [0,1] as default !
            lr = gaussian_blur2d(hr/(self.args.rgb_range*1.0), [kx, ky], [sx, sy])*self.args.rgb_range
            lr = quantize(lr, self.args.rgb_range)
            return lr

        res_lr = []
        for img in hr:
            img = np.uint8((img.permute(1, 2, 0)*255).numpy())  # tensor to numpy
            if self.blur:
                lr = self.AddBlur(img)
            else:
               lr = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(img), h))), size=size)
            lr = self.transform(lr)  # pil img to tensor between 0 and 1
            res_lr.append(lr)
        res_lr = torch.stack(res_lr)
        return res_lr
