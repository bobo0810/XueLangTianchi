# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
class DefaultConfig():
    env = 'Resnet152_XueLang_'  # visdom 环境的名字
    # 使用的模型，名字必须与models/__init__.py中的名字一致


    # 目前支持的网络
    # model = 'VGG16_bo' #vgg16不如vgg19好
    # model = 'VGG19_bo'
    # model = 'ResNet18_bo'
    model = 'ResNet152_bo'





    #数据集地址
    data_root='/home/bobo/windowsPycharmProject/Resnet152/data/train_resnet152_paert123.txt'   #数据集地址
    test_data_root='/home/bobo/data/xuelang_round1_test_a_20180709' #官方测试集地址
    # test_data_root ='/home/bobo/data/xuelang_shoudong' #手动测试集地址

    # 保存模型
    checkpoint_root='/home/bobo/windowsPycharmProject/Resnet152/checkpoint/'  #存储模型的路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载





    # 训练网络相关参数
    fixed_weight=False  #若True,则只更新部分层参数，为False则更新全部参数（采用pytorch提供的预训练好的模型,判断固定权重）
    batch_size = 16  # 训练集的batch size32
    val_batch_size= 16 # 验证集的batch size32

    result_file='result.csv'  #将测试结果写入csv文件中
    ratio=0.8  #训练集与验证集划分比率（接近1的数）
    use_gpu = True  # user GPU or not
    num_workers = 4  #  加载数据时的线程数
    print_freq =8  # 训练时，每N个batch显示

    max_epoch = 10000
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

#初始化该类的一个对象
opt=DefaultConfig()