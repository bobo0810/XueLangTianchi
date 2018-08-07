# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
import datetime
import os
class DefaultConfig():
    env = 'XueLang_'  # visdom 环境的名字
    # 使用的模型，名字必须与models/__init__.py中的名字一致


    # 目前支持的网络
    # model = 'VGG16_bo'
    #model = 'VGG19_bo'
    # model = 'ResNet18_bo'
    model = 'ResNet152_bo'



    # 数据预处理地址
    #预测图像的左上角切分坐标（420 一半步长）
    CropCoordinates='0 0 420 0 210 420 0 420 420 0 630 420 0 840 420 0 1050 420 0 1260 420 0 1470 420 0 1680 420 0 1890 420 0 2100 420 0 2140 420 210 0 420 210 210 420 210 420 420 210 630 420 210 840 420 210 1050 420 210 1260 420 210 1470 420 210 1680 420 210 1890 420 210 2100 420 210 2140 420 420 0 420 420 210 420 420 420 420 420 630 420 420 840 420 420 1050 420 420 1260 420 420 1470 420 420 1680 420 420 1890 420 420 2100 420 420 2140 420 630 0 420 630 210 420 630 420 420 630 630 420 630 840 420 630 1050 420 630 1260 420 630 1470 420 630 1680 420 630 1890 420 630 2100 420 630 2140 420 840 0 420 840 210 420 840 420 420 840 630 420 840 840 420 840 1050 420 840 1260 420 840 1470 420 840 1680 420 840 1890 420 840 2100 420 840 2140 420 1050 0 420 1050 210 420 1050 420 420 1050 630 420 1050 840 420 1050 1050 420 1050 1260 420 1050 1470 420 1050 1680 420 1050 1890 420 1050 2100 420 1050 2140 420 1260 0 420 1260 210 420 1260 420 420 1260 630 420 1260 840 420 1260 1050 420 1260 1260 420 1260 1470 420 1260 1680 420 1260 1890 420 1260 2100 420 1260 2140 420 1470 0 420 1470 210 420 1470 420 420 1470 630 420 1470 840 420 1470 1050 420 1470 1260 420 1470 1470 420 1470 1680 420 1470 1890 420 1470 2100 420 1470 2140 420 '

    xuelangtrainpart123_dataset='../../data/xuelangtrainpart123_dataset.txt' #数据集
    xuelangtrainpart123='../../data/xuelangtrainpart123.txt'  #计算IOU

    #数据集地址
    data_root=''  #训练集由xuelangtrainpart123_dataset.txt文件读取，故这里不再制定路径
    # test_data_root='../../data/xuelang_round1_test_a_20180709' #官方测试集地址1
    test_data_root = '../../data/xuelang_round1_test_b'  # 官方测试集地址2
    val_data_root = '/home/bobo/data/finaVal3'  # part123抽取的图片，作为验证集来算AUC





    # 保存模型
    root = os.path.abspath(os.path.join(os.path.dirname(__file__))) + '/'
    checkpoint_root=root+'../checkpoint/'  #存储模型的路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载（用于训练）
    # load_model_path = checkpoint_root+'ResNet152_bo.pth'




    # 训练网络相关参数
    fixed_weight=False  #若True,则只更新部分层参数，为False则更新全部参数（采用pytorch提供的预训练好的模型,判断固定权重）
    batch_size = 64 # 训练集的batch size
    val_batch_size= 1 # 验证集的batch size(part123抽取的图像,只支持batch=1)

    result_file="../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"  #将测试结果写入csv文件中

    use_gpu = True  # user GPU or not
    use_multi_gpu=True  #是否使用多GPU并行，若为True,则占用两块GPU
    num_workers = 4  #  加载数据时的线程数
    print_freq =8  # 训练时，每N个batch显示
    val_after_epoch=8  #当epoch到达某个epoch之后，再进行验证

    max_epoch = 90
    current_epoch=0
    lr = 0.001 # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数


#初始化该类的一个对象
opt=DefaultConfig()