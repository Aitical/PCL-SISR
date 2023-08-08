import os.path
import random
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageFilter
import numpy as np
import cv2
import math
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image
import PIL


class AlignedDataset(Dataset):

    def __init__(self, root, file_list,transform):
        super().__init__()
        self.root = root
        with open(file_list, 'r') as f:
            data = [i.strip() for i in f.readlines()]

        self.file_list = data
        self.transform = transform

    def AddNoise(self,img): # noise
        # if random.random() > 0.9: #
        #     return img
        self.sigma = np.random.randint(1, 25)
        img_tensor = torch.from_numpy(np.array(img)).float()
        noise = torch.randn(img_tensor.size()).mul_(self.sigma/1.0)

        noiseimg = torch.clamp(noise+img_tensor,0,255)
        return Image.fromarray(np.uint8(noiseimg.numpy()))

    def AddBlur(self,img): # gaussian blur or motion blur
        # if random.random() > 0.9: #
        #     return img
        img = np.array(img)
        if random.random() > 0.35: ##gaussian blur
            blursize = random.randint(1, 12)*2 + 1 ##3,5,7,9,11,13,15
            blursigma = random.randint(3, 20)
            img = cv2.GaussianBlur(img, (blursize,blursize), blursigma*1.0)
        else: #motion blur
            M = random.randint(1,32)
            KName = './data/MotionBlurKernel/m_%02d.mat' % M
            k = loadmat(KName)['kernel']
            k = k.astype(np.float32)
            k /= np.sum(k)
            img = cv2.filter2D(img,-1,k)
        return Image.fromarray(img)

    def AddDownSample(self,img): # downsampling
        # if random.random() > 0.95: #
        #     return img
        sampler = random.randint(10, 200)*1.0
        img = img.resize((int(512/sampler), int(512/sampler)), Image.BICUBIC)
        return img

    def AddJPEG(self,img): # JPEG compression
        # if random.random() > 0.6: #
        #     return img
        imQ = random.randint(50, 95)
        img = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),imQ] # (0,100),higher is better,default is 95
        _, encA = cv2.imencode('.jpg',img,encode_param)
        img = cv2.imdecode(encA,1)
        return Image.fromarray(img)

    def AddUpSample(self,img):
        return img.resize((512, 512), Image.BICUBIC)

    def __getitem__(self, index): #

        AB_path = self.file_list[index]
        imgs = Image.open(AB_path).convert('RGB')
        # #
        hr = imgs.resize((512, 512), Image.BICUBIC)
        lr = imgs.resize((512, 512), Image.BICUBIC)
        lr = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(lr)))))

        lr = self.transform(lr)
        hr = self.transform(hr)

        return lr, hr

    def get_part_location(self, landmarkpath, imgname, downscale=1):
        Landmarks = []
        with open(os.path.join(landmarkpath,imgname+'.txt'),'r') as f:
            for line in f:
                tmp = [np.float(i) for i in line.split(' ') if i != '\n']
                Landmarks.append(tmp)
        Landmarks = np.array(Landmarks)/downscale # 512 * 512

        Map_LE = list(np.hstack((range(17,22), range(36,42))))
        Map_RE = list(np.hstack((range(22,27), range(42,48))))
        Map_NO = list(range(29,36))
        Map_MO = list(range(48,68))
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))

        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
        return Location_LE, Location_RE, Location_NO, Location_MO

    def __len__(self): #
        return len(self.file_list)

    def name(self):
        return 'AlignedDataset'