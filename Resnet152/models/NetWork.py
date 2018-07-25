# -*- coding:utf-8 -*-
# power by Mr.Li
from Resnet152.utils.config  import opt
from torch import nn
import  time
import torch as t
import torch.nn.functional as F
from torchvision.models import vgg16_bn,vgg19_bn
import math  as math
class VGG16_bo(nn.Module):
    '''
    定义网络
    '''
    def __init__(self):
        super(VGG16_bo,self).__init__()
        model=vgg16_bn(pretrained=True)

        # 设置网络名称
        self.moduel_name=str("VGG16_bo")

        #提取特征层，权重为预训练权重
        self.features=model.features
        # # 固定权重
        for param in  self.features.parameters():
            param.requires_grad=False

        # 分类层
        self.classifier=nn.Sequential(
            t.nn.Linear(25088, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.5),
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.5),
            t.nn.Linear(4096,2)
        )
        # 仅对分类层初始化
        self._initialize_weights()

    def forward(self,x):
        # 前向传播
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # 得到输出，经过sigmoid 归一化到0-1之间
        # x = F.softmax(x) #损失函数若使用交叉熵，则交叉熵自带softmax，不需要加softmax层
        x = x.view(-1, 2)
        return x

    def _initialize_weights(self):
        '''
        初始化网络权重(仅对分类层初始化)
        '''
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def save(self, name=None):
        '''
        设置存储模型的路径
        '''
        # 保存训练集每次epoch之后的模型
        if name is None:
            prefix = opt.checkpoint_root + self.moduel_name
            name=prefix+'.pth'
        # 保存在验证集上表现最好的模型
        else:
            prefix =opt.checkpoint_root ++'auc'+name+self.moduel_name + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


class VGG19_bo(nn.Module):
    '''
    定义网络
    '''
    def __init__(self):
        super(VGG19_bo,self).__init__()
        model=vgg19_bn(pretrained=True)
        # 设置网络名称
        self.moduel_name=str("VGG19_bo")

        #提取特征层，权重为预训练权重
        self.features=model.features
        # # 固定权重
        # for param in  self.features.parameters():
        #     param.requires_grad=False

        # 分类层
        self.classifier=nn.Sequential(
            t.nn.Linear(86528, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.5),
            #取消一层全连接
            # t.nn.Linear(4096, 4096),
            # t.nn.ReLU(),
            # t.nn.Dropout(p=0.5),
            t.nn.Linear(4096,2)
        )
        # 仅对分类层初始化
        self._initialize_weights()

    def forward(self,x):
        # 前向传播
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # 得到输出，经过sigmoid 归一化到0-1之间
        # x = F.softmax(x) #损失函数若使用交叉熵，则交叉熵自带softmax，不需要加softmax层
        x = x.view(-1, 2)
        return x

    def _initialize_weights(self):
        '''
        初始化网络权重(仅对分类层初始化)
        '''
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def save(self, name=None):
        '''
        设置存储模型的路径
        '''
        # 保存训练集每次epoch之后的模型
        if name is None:
            prefix = opt.checkpoint_root + self.moduel_name
            name=prefix+'.pth'
        # 保存在验证集上表现最好的模型
        else:
            prefix =opt.checkpoint_root + 'auc'+name+self.moduel_name
            name = prefix + '.pth'
        t.save(self.state_dict(), name)
        return name

def test():
    from torch.autograd import Variable
    model=VGG19_bo()
    img=t.rand(2,3,420,420)
    img=Variable(img)
    output=model(img)
    print(output.size())

if __name__ == '__main__':
    test()