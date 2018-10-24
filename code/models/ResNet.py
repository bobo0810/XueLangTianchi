# -*- coding:utf-8 -*-
# power by Mr.Li
from xuelang.code.utils.config import opt
from torch import nn
import  time
import torch as t
from torchvision.models import resnet18,resnet152
from torch.autograd import Variable
class ResNet18_bo(nn.Module):
    '''
    定义ResNet18网络
    '''
    def __init__(self):
        super(ResNet18_bo,self).__init__()

        # 设置网络名称,保存模型时命名使用
        self.moduel_name=str("ResNet18_bo")
        # 加载预训练好的网络权重
        model = resnet18(pretrained=True)

        # 固定权重   nn.Module有成员函数parameters()
        if opt.fixed_weight:
            for param in model.parameters():
                param.requires_grad = False
        # 结论：self.model_bo只有修改过的新层（即最后两层全连接层）的值为True
        # 替换最后一层全连接层
        # 新层默认requires_grad=True
        # resnet18中有self.fc，作为前向过程的最后一层
        # （修改输入图像大小，可通过报错信息来调整下面参数）
        model.fc = nn.Linear(512, 2)   #224:512   420:32768
        # 此时self.model_bo的权重为预训练权重，修改的新层（全连接层）权重为自动初始化的
        self.model_bo=model


        # # 加载模型的参数
        # params=self.model_bo.state_dict()
        # print(params['fc.weight'])
        #手动初始化fc层
        self._initialize_weights()
        # # 查看初始化之后的fc层参数
        # params2=self.model_bo.state_dict()
        # print(params2['fc.weight'])
        ## 结论：手动初始化fc层生效


    def forward(self,x):
        # 前向传播
        x=self.model_bo(x)
        # 得到输出，经过sigmoid 归一化到0-1之间
        # x = F.softmax(x) #损失函数若使用交叉熵，则交叉熵自带softmax，不需要加softmax层
        x = x.view(-1, 2)
        return x

    def _initialize_weights(self):
        '''
        初始化网络权重
        '''
        m=self.model_bo.fc
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def save(self, name=None):
        '''
        设置存储模型的路径
        '''
        # 保存训练集每次epoch之后的模型
        if name is None:
            prefix = opt.checkpoint_root + self.moduel_name + "_"
            # name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            name=prefix+'resnet18.pth'
        # 保存在验证集上表现最好的模型
        else:
            prefix =opt.checkpoint_root + 'val_best_loss'+self.moduel_name + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name



class ResNet152_bo(nn.Module):
    '''
    定义ResNet152网络
    '''
    def __init__(self):
        super(ResNet152_bo,self).__init__()

        # 设置网络名称,保存模型时命名使用
        self.moduel_name=str("ResNet152_bo")
        # 加载预训练好的网络权重
        model = resnet152(pretrained=True)

        # 固定权重   nn.Module有成员函数parameters()
        if opt.fixed_weight:
            for param in model.parameters():
                param.requires_grad = False
        # 结论：self.model_bo只有修改过的新层（即最后两层全连接层）的值为True
        # 替换最后一层全连接层
        # 新层默认requires_grad=True
        # resnet152中有self.fc，作为前向过程的最后一层
        # （修改输入图像大小，可通过报错信息来调整下面参数）
        model.fc = nn.Linear(2048, 2)   #420:131072    224:2048
        # 此时self.model_bo的权重为预训练权重，修改的新层（全连接层）权重为自动初始化的
        self.model_bo=model
        #手动初始化fc层
        self._initialize_weights()



    def forward(self,x):
        # 前向传播
        x=self.model_bo(x)
        # x = F.softmax(x) #损失函数若使用交叉熵，则交叉熵自带softmax，不需要加softmax层
        x = x.view(-1, 2)
        return x

    def _initialize_weights(self):
        '''
        初始化网络权重
        '''
        m=self.model_bo.fc
        if isinstance(m, nn.Linear):
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
            prefix =opt.checkpoint_root + 'auc'+name+self.moduel_name + "_"
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
def test():

    model=ResNet18_bo()
    params=model.state_dict()
    img=t.rand(2,3,224,224)
    img=Variable(img)
    output=model(img)
    print(output.size())

if __name__ == '__main__':
    test()