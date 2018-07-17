# -*- coding:utf-8 -*-
# power by Mr.Li
# 设置默认参数
class DefaultConfig():
    env = 'Xuelang_YOLOv1'  # visdom 环境的名字
    file_root = ''  #VOC2012的训练集
    test_root = ''   #VOC2007的测试集
    train_Annotations = "/home/bobo/Download/xuelang_round1_train_part3_20180709" #Download只有part2和part3数据集

    voc_2007test='/home/bobo/windowsPycharmProject/XuelangYOLOv1ByBobo/data/xuelang_part2_3.txt'
    voc_2012train='/home/bobo/windowsPycharmProject/XuelangYOLOv1ByBobo/data/xuelang_part2_3.txt'


    test_img_dir='/home/bobo/data/xuelang_shoudong'
    result_img_dir='/home/bobo/data/xuelang_shoudong_result'

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data  加载数据时的线程
    print_freq = 20  # print info every N batch


    best_test_loss_model_path= '/home/bobo/windowsPycharmProject/XuelangYOLOv1ByBobo/checkpoint/best_val.pth'
    current_epoch_model_path='/home/bobo/windowsPycharmProject/XuelangYOLOv1ByBobo/checkpoint/new.pth'
    load_model_path = current_epoch_model_path  # 加载预训练的模型的路径，为None代表不加载


    num_epochs = 120   #训练的epoch次数
    learning_rate = 0.0001  # initial learning rate
    # lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    momentum=0.9
    weight_decay =5e-4  # 损失函数
    # 瑕疵的类别 共48类
    VOC_CLASSES=(
        '吊纬', '扎洞', '正常', '毛斑', '修印', '剪洞', '厚薄段', '吊弓', '吊经', '回边', '嵌结', '弓纱',
        '愣断', '扎梳', '扎纱', '擦伤', '擦毛', '擦洞', '楞断', '毛洞', '毛粒', '污渍', '油渍', '破洞', '破边',
        '粗纱', '紧纱', '纬粗纱', '线印', '织入', '织稀', '经粗纱', '经跳花', '结洞', '缺纬', '缺经',
        '耳朵', '蒸呢印', '跳花', '边扎洞', '边白印', '边缺纬', '边针眼', '黄渍', '厚段', '夹码', '明嵌线', '边缺经'
    )


 #初始化该类的一个对象
opt=DefaultConfig()