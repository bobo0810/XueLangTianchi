#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
import os
from PIL import  Image
from torch.utils import data
import numpy as np
from torchvision import  transforms as T
import torch as t
from Resnet152.utils.config import opt
import cv2
import random
from math import ceil
import numpy as np
import glob

class XueLangDataSet(data.Dataset):
    '''
    主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
    '''
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test=test  #状态
        self.train = train
        self.root=root  #数据集路径
        self.image_size = 420 #最后图像规整
        self.mean = (123, 117, 104)  # ImageNet数据集的RGB均值
        # 存储图像
        self.imgs=[]
        # 存储对应标签
        self.label = []

        imgs=[]
        label=[]

        if self.test:
            pass
        else:
            with open(self.root) as f:
                lines = f.readlines()
            # 打乱数据集顺序
            lines_copy = np.copy(lines).tolist()
            random.shuffle(lines_copy)
            # 遍历每一行
            for line in lines_copy:
                splited = line.strip().split()
                imgs.append(splited[0]+" "+splited[1])
                label.append(splited[2])
            # 数据集的图片总数
            imgs_num=len(label)
        # 如果是测试集就直接用
        if self.test:
            # 读取测试集文件夹下所有图像
            imgs_test = glob.glob(os.path.join(self.root, '*.jpg'))
            self.imgs = imgs_test


        elif train:  # 训练集
            self.imgs = imgs[:int(opt.ratio * imgs_num)]
            self.label = label[:int(opt.ratio * imgs_num)]


        else:  # 验证集
            self.imgs = imgs[int(opt.ratio * imgs_num):]
            self.label = label[int(opt.ratio * imgs_num):]


        # 对图像进行转化(若未指定转化，则执行默认操作)
        if transforms is None:
            normalize = T.Normalize(mean=[0, 0, 0],
                                    std=[1, 1, 1])

            if self.test or not train: #测试集和验证集
                self.transforms = T.Compose([
                    # T.Resize(224), #尺度放缩到224
                    # T.CenterCrop(224),#中心裁剪到224
                    T.ToTensor(),
                    # normalize
                ])
            else:   #训练集
                self.transforms = T.Compose([
                    # T.Resize(224), #尺度放缩到224
                    # T.RandomResizedCrop(224), #随机裁剪到224
                    # T.RandomHorizontalFlip(), #水平翻转
                    T.ToTensor(),
                    # normalize
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path_origin = self.imgs[index]
        if self.test:
            pass
        else:
            label=int(self.label[index][0])

        # 读取图像
        img = cv2.imread(img_path_origin)

        # 测试集
        if self.test:
            img = self.BGR2RGB(img)
            img = cv2.resize(img, (self.image_size, self.image_size))
            data_img = self.transforms(img)
            img_name = img_path_origin.split('/')[-1]
            return data_img,img_name
        # 训练集和验证集
        else:

            # 如果为训练集,进行数据增强
            if self.train:
                # 随机翻转
                img = self.random_flip(img)
                # 固定住高度，以0.6-1.4伸缩宽度，做图像形变,会改变图像大小
                img = self.randomScale(img)
                # 随机模糊
                img = self.randomBlur(img)
                # 随机亮度
                img = self.RandomBrightness(img)
                # 随机色调
                img = self.RandomHue(img)
                # 随机饱和度
                img = self.RandomSaturation(img)
                # 随机平移转换,原图像会平移到一角，剩余内容空白
                img = self.randomShift(img)
            img = self.BGR2RGB(img)  # 因为pytorch自身提供的预训练好的模型期望的输入是RGB
            img = cv2.resize(img, (self.image_size, self.image_size))

            #对图片进行转化
            data_img = self.transforms(img)
            return data_img, label






    def __len__(self):
        '''
        返回数据集的图片总数
        '''
        return len(self.imgs)


    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        '''
         随机模糊
        '''
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr):
        # 平移变换

        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]
            return after_shfit_image
        return bgr

    def randomScale(self,bgr):
        #固定住高度，以0.6-1.4伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.6,1.4)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            return bgr
        return bgr

    def randomCrop(self,bgr):
        if random.random() < 0.5:
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped
        return bgr

    def random_flip(self, im):
        '''
        随机翻转
        '''
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            return im_lr
        return im












