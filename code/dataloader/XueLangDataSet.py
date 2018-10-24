#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
import os
from torch.utils import data
from torchvision import  transforms as T
import torch as t
from xuelang.code.utils.config import opt
import cv2
import random
from xuelang.code.utils.calculateIOU import judge_much_IOU
import numpy as np

class XueLangDataSet(data.Dataset):
    '''
    主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
    '''
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test=test  #状态
        self.train = train
        self.root=root  #数据集路径
        self.image_size = 224  #最后图像规整

        self.img_dataset=[]


        # 读取文件夹下所有图像
        if root!='':
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
        else:
            imgs=[]


        # 测试集
        if self.test:
            # 返回新的list
            imgs_origin = np.copy(imgs)
        elif train:
            list_file = opt.xuelangtrainpart123_dataset
            root=os.path.abspath(os.path.join(os.path.dirname(__file__)))+'/'
            with open(root+list_file) as f:
                self.img_dataset=f.readlines()
            random.shuffle(self.img_dataset)
        else:  #验证集
            imgs_origin = np.copy(imgs)

        # 如果是测试集就直接用
        if self.test:
            self.imgs_origin = imgs_origin
        elif train:#训练集
            self.imgs_origin=self.img_dataset
        else:#验证集
            self.imgs_origin = imgs_origin
        # 对图像进行转化(若未指定转化，则执行默认操作)
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.test or not train: #测试集和验证集
                self.transforms = T.Compose([
                    T.ToTensor(),
                    normalize
                ])
            else:   #训练集
                self.transforms = T.Compose([

                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''



        # 测试集
        if self.test:
            img_path_origin = self.imgs_origin[index]
            # 读取图像
            img = cv2.imread(img_path_origin)

            # 如果是测试集，则切分原图
            # 放原图
            img_copy = np.copy(img)
            img_copy = self.BGR2RGB(img_copy)
            img_copy = cv2.resize(img_copy, (self.image_size, self.image_size))
            data_img_copy = self.transforms(img_copy)
            # 扩展一维
            data_img_copy= t.unsqueeze(data_img_copy, dim=0)
            # list_file = opt.CropCoordinates
            # with open(list_file) as f:
            #     lines = f.readlines()
            # # 按照空格切分一行
            # splited = lines[0].strip().split()
            splited =opt.CropCoordinates.strip().split()
            for i in range(len(splited)):
                if i % 3 == 0:
                    x,y,length_number=int(splited[i]),int(splited[i + 1]),int(splited[i + 2])
                    Img_test = img[x:(x + length_number), y:(y + length_number), ]
                    Img_test = self.BGR2RGB(Img_test)
                    Img_test = cv2.resize(Img_test, (self.image_size, self.image_size))
                    Img_test_transforms = self.transforms(Img_test)
                    # 扩展一维
                    Img_test_transforms=t.unsqueeze(Img_test_transforms, dim=0)
                    data_img_copy = t.cat((data_img_copy, Img_test_transforms), 0)

            img_name = img_path_origin.split('/')[-1]
            return data_img_copy,img_name

        # 如果为训练集,进行数据增强
        elif  self.train:
            img_info = self.img_dataset[index].split(',')
            img_path_origin = img_info[0]
            img_label = img_info[1]
            # 读取图像
            # root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
            img = cv2.imread(img_path_origin)

            # 得到文件名
            img_name = img_path_origin.split('/')[-1]
            crop_size=420  #用于 有瑕疵图的裁剪
            # 即为无瑕疵原图  （裁剪420 标签为0）
            if 'origin' in img_label   and  'J01' not in img_label :
                #30个epoch之前，网络按照无瑕疵原图训练。之后，网络将按照 0.5概率按照原图训练，0.5概率进行裁剪后再训练。
                if opt.current_epoch>30:
                    if random.random()<0.5:
                        x_crop = random.randint(0, 1920)
                        y_crop = random.randint(0, 2560)
                        if (x_crop + 420) > 1920:
                            x_crop = 1920 - 420 - 1
                        if (y_crop + 420) > 2560:
                            y_crop = 2560 - 420 - 1
                        img = img[x_crop:(x_crop + 420), y_crop:(y_crop + 420), ]
                else:
                    pass

                # 随机模糊
                img = self.randomBlur(img)
                # 随机亮度
                img = self.RandomBrightness(img)
                # 随机色调
                img = self.RandomHue(img)
                # 随机饱和度
                img = self.RandomSaturation(img)
                #随机翻转
                img = self.random_flip(img)

                img = self.BGR2RGB(img)  # 因为pytorch自身提供的预训练好的模型期望的输入是RGB
                img = cv2.resize(img, (self.image_size, self.image_size))
                # 对图片进行转化
                data_img = self.transforms(img)
                # 设置标签
                label = 0  # 无瑕疵
                return data_img, label

            # 从有瑕疵中裁出无瑕疵的（标签为0）
            if 'J01' in img_label and 'origin' in img_label:
                root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
                list_file = root+opt.xuelangtrainpart123
                with open(list_file) as f:
                    lines = f.readlines()

                # 遍历每一行
                for line in lines:
                    # bbox  写入所有物体的坐标值
                    box = []
                    result=0
                    # 要求名字必须是 有瑕疵原图的 原名
                    if img_name in line:
                        # 按照空格切分一行
                        splited = line.strip().split(',')
                        # 赋值一张图的物体总数
                        num_object = int(splited[1])
                        for i in range(num_object):
                            # 4个坐标类型为字符类型
                            xmin = splited[2 + 5 * i]
                            ymin = splited[3 + 5 * i]
                            xmax = splited[4 + 5 * i]
                            ymax = splited[5 + 5 * i]
                            box.append([xmin, ymin, xmax, ymax])
                        # 以上是拿到了bbox的坐标，可能是好几个bbox，以下是裁剪
                        h, w = 1920, 2560
                        size = 420  # 裁剪的图片的大小
                        # 直到随机裁剪到一张IOU为0的图片即可break
                        while True:
                            x_begin = random.randint(0, w - size - 1)
                            y_begin = random.randint(0, h - size - 1)
                            # 做一个IOU的判断
                            this_box_list = [x_begin, y_begin, x_begin + size, y_begin + size]
                            result = judge_much_IOU.judge_much_IOU(box, this_box_list)
                            # 即 1:表示 裁剪与原bbox没有交集
                            if result is 1:
                                # 裁剪
                                img = img[y_begin:(y_begin + 420), x_begin:(x_begin + 420), ]
                                break
                    if result is 1:
                        break


                # 随机模糊
                img = self.randomBlur(img)
                # 随机亮度
                img = self.RandomBrightness(img)
                # 随机色调
                img = self.RandomHue(img)
                # 随机饱和度
                img = self.RandomSaturation(img)
                # 随机翻转
                img=self.random_flip(img)
                img = self.BGR2RGB(img)  # 因为pytorch自身提供的预训练好的模型期望的输入是RGB
                img = cv2.resize(img, (self.image_size, self.image_size))
                # 对图片进行转化
                data_img = self.transforms(img)
                # 设置标签
                label = 0  # 无瑕疵
                return data_img, label

            # 即为有瑕疵原图（用于裁剪bbox   标签为1）
            if 'J01' in img_label  and 'origin' not in img_label:

                root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
                list_file = root+opt.xuelangtrainpart123
                with open(list_file) as f:
                    lines = f.readlines()
                # 改为原图中的名字，以便根据原名找到对应bbox
                img_name = img_name[0:23] + '.jpg'
                # 遍历每一行
                for line in lines:
                    # 要求名字必须是 有瑕疵原图的 原名
                    if img_name in line:
                        # 按照空格切分一行
                        splited = line.strip().split(',')
                        # 赋值一张图的物体总数
                        num_object = int(splited[1])

                        # 随机挑选一个bbox,进行包围它
                        i = random.randint(0, num_object - 1)
                        # 4个坐标类型为字符类型
                        xmin = splited[2 + 5 * i]
                        ymin = splited[3 + 5 * i]
                        xmax = splited[4 + 5 * i]
                        ymax = splited[5 + 5 * i]
                        w, h = int(xmax) - int(xmin), int(ymax) - int(ymin)
                        # 都小于420
                        if w < crop_size and h < crop_size:
                            # 中心点坐标
                            cent_coord = np.array(
                                [int(ymin) + (int(ymax) - int(ymin)) // 2, int(xmin) + (int(xmax) - int(xmin)) // 2])
                            # 拿到图像左上角坐标
                            # 判断是否超出原图范围
                            if (cent_coord[0] - (crop_size // 2)) < 0:
                                y1 = 0
                            else:
                                y1 = cent_coord[0] - (crop_size // 2)
                            y2 = int(ymin)

                            if (cent_coord[1] - (crop_size // 2)) < 0:
                                x1 = 0
                            else:
                                x1 = cent_coord[1] - (crop_size // 2)
                            x2 = int(xmin)
                            # 随机在这个范围内产生一个左上角坐标   宽高都为crop_size  420
                            y_min_random = random.randint(y1, y2)
                            x_min_random = random.randint(x1, x2)
                            if y_min_random + crop_size > 1920:
                                y_min_random = 1920 - crop_size
                                y_max_random = 1920
                            else:
                                y_max_random = y_min_random + crop_size

                            if x_min_random + crop_size > 2560:
                                x_min_random = 2560 - crop_size
                                x_max_random = 2560
                            else:
                                x_max_random = x_min_random + crop_size
                            # 裁剪目标
                            img = img[y_min_random:y_max_random, x_min_random:x_max_random, ]

                        # 都大于420
                        if w > crop_size and h > crop_size:
                            # 一半几率用原bbox
                            if random.random() < 0.5:
                                img = img[int(ymin):int(ymax), int(xmin):int(xmax), ]
                            # 一半几率在原bbox裁剪420
                            else:
                                # 随机在这个范围内产生一个左上角坐标   宽高都为crop_size  420
                                y_min_random = random.randint(int(ymin), int(ymax) - crop_size)
                                x_min_random = random.randint(int(xmin), int(xmax) - crop_size)
                                # 裁剪目标
                                img = img[y_min_random:y_min_random + crop_size,
                                         x_min_random:x_min_random + crop_size, ]

                        # w<420  h>420
                        if w < crop_size and h > crop_size:
                            # 拿到图像左上角坐标
                            if (int(xmax) - crop_size) < 0:
                                x1 = 0
                            else:
                                x1 = int(xmax) - crop_size
                            x2 = int(xmin)

                            y1 = int(ymin)
                            y2 = int(ymax) - crop_size

                            # 随机在这个范围内产生一个左上角坐标   宽高都为crop_size  420
                            y_min_random = random.randint(y1, y2)
                            x_min_random = random.randint(x1, x2)

                            y_max_random = y_min_random + crop_size

                            if x_min_random + crop_size > 2560:
                                x_min_random = 2560 - crop_size
                                x_max_random = 2560
                            else:
                                x_max_random = x_min_random + crop_size
                            # 裁剪目标
                            img = img[y_min_random:y_max_random, x_min_random:x_max_random, ]

                        # w>420  h<420
                        if w > crop_size and h < crop_size:
                            # 拿到图像左上角坐标
                            if (int(ymax) - crop_size) < 0:
                                y1 = 0
                            else:
                                y1 = int(ymax) - crop_size
                            y2 = int(ymin)

                            x1 = int(xmin)
                            x2 = int(xmax) - crop_size

                            # 随机在这个范围内产生一个左上角坐标   宽高都为crop_size  420
                            y_min_random = random.randint(y1, y2)
                            x_min_random = random.randint(x1, x2)

                            x_max_random = x_min_random + crop_size

                            if y_min_random + crop_size > 1920:
                                y_min_random = 1920 - crop_size
                                y_max_random = 1920
                            else:
                                y_max_random = y_min_random + crop_size
                            # 裁剪目标
                            img = img[y_min_random:y_max_random, x_min_random:x_max_random, ]
                        break

                # 随机模糊
                img = self.randomBlur(img)
                # 随机亮度
                img = self.RandomBrightness(img)
                # 随机色调
                img = self.RandomHue(img)
                # 随机饱和度
                img = self.RandomSaturation(img)
                # 随机翻转
                img=self.random_flip(img)

                img = self.BGR2RGB(img)  # 因为pytorch自身提供的预训练好的模型期望的输入是RGB
                img = cv2.resize(img, (self.image_size, self.image_size))
                # 对图片进行转化
                data_img = self.transforms(img)
                # 设置标签
                label = 1  # 有瑕疵
                return data_img, label





        #验证集（使用part123抽取的正负样本1:1来计算AUC）
        else:
            img_path_origin = self.imgs_origin[index]
            # 读取图像
            img = cv2.imread(img_path_origin)

            # 切分原图
            # 放原图
            img_copy = np.copy(img)
            img_copy = self.BGR2RGB(img_copy)
            img_copy = cv2.resize(img_copy, (self.image_size, self.image_size))
            data_img_copy = self.transforms(img_copy)
            # 扩展一维
            data_img_copy = t.unsqueeze(data_img_copy, dim=0)


            splited = opt.CropCoordinates.strip().split()
            for i in range(len(splited)):
                if i % 3 == 0:
                    x, y, length_number = int(splited[i]), int(splited[i + 1]), int(splited[i + 2])
                    Img_test = img[x:(x + length_number), y:(y + length_number), ]
                    Img_test = self.BGR2RGB(Img_test)
                    Img_test = cv2.resize(Img_test, (self.image_size, self.image_size))
                    Img_test_transforms = self.transforms(Img_test)
                    # 扩展一维
                    Img_test_transforms = t.unsqueeze(Img_test_transforms, dim=0)
                    data_img_copy = t.cat((data_img_copy, Img_test_transforms), 0)

            img_name = img_path_origin.split('/')[-1]
            if 'origin' in img_name:
                label = 0  # 无瑕疵
            else:
                label = 1  # 有瑕疵
            return data_img_copy, label








    def __len__(self):
        '''
        返回数据集的图片总数
        '''
        return len(self.imgs_origin)


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












