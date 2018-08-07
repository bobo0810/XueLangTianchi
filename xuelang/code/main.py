#!/usr/bin/python
# -*- coding:utf-8 -*-
# power by Mr.Li
from xuelang.code.utils.config import opt
import torch.backends.cudnn as cudnn
import torch as t
import xuelang.code.models as models
from xuelang.code.dataloader.XueLangDataSet import XueLangDataSet   #加载转换后的数据集
from torch.utils.data import DataLoader  #数据加载器
from torch.autograd import Variable
from torchnet import meter  #仪表  用来显示loss等图形
from xuelang.code.utils.visualize import Visualizer  #可视化visdom
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from xuelang.code.predata.xml_txt_IOU import xml_txt_IOU
from xuelang.code.predata.xml_txt_dataset import xml_txt_dataset

def train(**kwargs):

    print("开始训练")
    # 定义一个网络模型对象
    # 通过config文件中模型名称来加载模型
    netWork = getattr(models, opt.model)()
    print('当前使用的模型为'+opt.model)

    # 定义可视化对象
    vis = Visualizer(opt.env+opt.model)

    # 先将模型加载到内存中，即CPU中
    map_location = lambda storage, loc: storage
    if opt.load_model_path:
        if opt.use_multi_gpu:
            # 加载多GPU模型
            pretained_model = t.load(opt.load_model_path)
            new_state_dict = OrderedDict()
            for k, v in pretained_model.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            netWork.load_state_dict(new_state_dict)
        else:
            # 加载单GPU模型
            netWork.load_state_dict(t.load(opt.load_model_path, map_location=map_location))
    if opt.use_multi_gpu:
        netWork=t.nn.DataParallel(netWork,device_ids=[0,1])
    # 将模型转到GPU
    if opt.use_gpu:
        netWork.cuda()
        cudnn.benchmark=True
    # step2: 加载数据
    train_data = XueLangDataSet(opt.data_root, train=True)
    # #train=False  test=False   则为验证集
    val_data = XueLangDataSet(opt.val_data_root,train=False)
    # 数据集加载器
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.val_batch_size, shuffle=True, num_workers=opt.num_workers)  #使用part123抽取的正负样本1:1来计算AUC
    # criterion 损失函数和optimizer优化器
    # 分类损失函数使用交叉熵
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    # 优化器使用Adam
    if opt.fixed_weight:
        # 选择固定部分权重参数
        if opt.model is 'ResNet18_bo'  or  opt.model is 'ResNet152_bo':
            # ResNet18_bo和ResNet152网络只更新最后的全连接层
            print(opt.model+'网络只更新最后的全连接层')
            optimizer = t.optim.Adam(netWork.model_bo.fc.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        if opt.model is 'VGG16_bo'  or opt.model is 'VGG19_bo':
            print(opt.model+'网络只更新分类层')
            optimizer = t.optim.Adam(netWork.classifier.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        # 更新全部参数
        print(opt.model+"参数全部更新")
        optimizer = t.optim.Adam(netWork.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)



    # 统计指标meters  仪表 显示损失的图形
    #计算所有数的平均数和标准差，来统计一个epoch中损失的平均值
    loss_meter=meter.AverageValueMeter()
    # 定义初始的loss
    previous_loss = 1e100
    # best_val_auc =0  #初始化验证集上最好的AUC
    for epoch in range(opt.max_epoch):

        opt.current_epoch=epoch
        #则更改数据集，并恢复学习率
        #30个epoch之前，网络按照无瑕疵原图训练。之后无瑕疵原图，对于网络将按照 0.5概率按照原图训练，0.5概率进行裁剪后再训练。
        if opt.current_epoch==30:
            print("到达30个epoch,恢复学习率为0.001,自动调整数据集")
            lr = 0.001
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 清空仪表信息
        loss_meter.reset()
        # 迭代数据集加载器
        for ii, (data_origin,label) in enumerate(train_dataloader):
            # 训练模型
            # input_img为模型输入图像
            input_img = Variable(data_origin)
            # label_img为对应标签
            label_img = Variable(label)
            # 将数据转到GPU
            if opt.use_gpu:
                input_img = input_img.cuda()
                label_img = label_img.cuda()
            # 优化器梯度清零
            optimizer.zero_grad()
            # 前向传播，得到网络产生的输出值label_output
            label_output = netWork(input_img)
            # 损失为交叉熵
            loss = criterion(label_output, label_img)
            # 反向传播  自动求梯度         loss进行反向传播
            loss.backward()
            # 更新优化器的可学习参数       optimizer优化器进行更新参数
            optimizer.step()
            # 更新仪表 并可视化
            loss_meter.add(loss.data[0])
            # 每print_freq次可视化loss
            if ii % opt.print_freq == opt.print_freq - 1:
                # plot是自定义的方法
                vis.plot('训练集loss', loss_meter.value()[0])
        if opt.use_multi_gpu:
            # 多GPU模型保存
            t.save(netWork.state_dict(), opt.checkpoint_root + opt.model + '.pth')
        else:
            # 单GPU模型保存
            netWork.save()
        print("第"+str(epoch)+"次epoch完成==========================")
        #  一个epoch之后保存模型
        # 当前时刻的一些信息
        vis.log("epoch:{epoch},lr:{lr},loss:{loss}".format(
            epoch=epoch, loss=loss_meter.value()[0], lr=lr))

        # 更新学习率  如果损失开始升高，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

        # 当epoch到达某个epoch之后，再进行验证
        if (epoch > opt.val_after_epoch and epoch % 2 == 0) or epoch > 20:
            # 在验证集上进行验证，保存在验证集上AUC最好的模型
            # 模型调整为验证模式
            netWork.eval()
            predict_label = []
            real_label = []
            for ii, (val_data_origin, val_label) in enumerate(val_dataloader):
                # 训练模型
                # input_img为模型输入图像
                val_input_img = Variable(val_data_origin, volatile=True)
                # label_img为对应标签
                val_label_img = Variable(val_label, volatile=True)
                # 将数据转到GPU
                if opt.use_gpu:
                    val_input_img = val_input_img.cuda()

                # 验证集batch为1,压缩第0维
                val_input_img = t.squeeze(val_input_img, dim=0)
                # 前向传播，得到网络产生的输出值label_output
                val_label = netWork(val_input_img)
                # 概率  通过softmax可得概率 一张图得到多个结果  shape:[X,2]
                val_label_score = t.nn.functional.softmax(val_label, dim=1)
                # score_order即为有瑕疵得分，从大到小排序
                score_order, _ = t.sort(val_label_score[:, 1].data, descending=True)
                # 取概率最大值作为最后结果
                score = score_order[0]
                if score == 1:
                    score = 0.999999
                if score < 0.000001:
                    score = 0.000001
                # 将每次的预测结果存入predict_label中
                predict_label.append(score)
                real_label.append(np.array(val_label_img.data)[0])
            # 过完一遍验证集，计算整个验证集上的AUC
            validation_auc_sklearn = roc_auc_score(real_label, predict_label)
            # 画出验证集的auc sklearn
            vis.plot('验证集(part123抽取的图像)的auc', validation_auc_sklearn)
            # 模型恢复为训练模式
            netWork.train()
            # 保存到目前为止 在验证集上的AUC最大的模型
            if best_val_auc < validation_auc_sklearn:
                best_val_auc = validation_auc_sklearn
                print('当前得到最好的验证集的AUC为  %.5f' % best_val_auc)
                if opt.use_multi_gpu:
                    # 多GPU模型保存
                    t.save(netWork.state_dict(),
                           opt.checkpoint_root + 'auc' + str(validation_auc_sklearn) + opt.model + '.pth')
                else:
                    # 单GPU模型保存
                    netWork.save(name=str(validation_auc_sklearn))
    print("============训练完毕=============")


def test(**kwargs):
    print("开始测试")
    # 定义一个网络模型对象
    # 通过config文件中模型名称来加载模型,并调整为验证模式
    netWork = getattr(models, opt.model)().eval()
    print('当前测试使用的模型为'+opt.model)
    # 先将模型加载到内存中，即CPU中
    map_location = lambda storage, loc: storage
    if opt.load_model_path:
        if opt.use_multi_gpu:
            # 加载多GPU模型
            pretained_model = t.load(opt.load_model_path)
            new_state_dict = OrderedDict()
            for k, v in pretained_model.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            netWork.load_state_dict(new_state_dict)
        else:
            netWork.load_state_dict(t.load(opt.load_model_path, map_location=map_location))
    # 将模型转到GPU
    if opt.use_multi_gpu:
        netWork = t.nn.DataParallel(netWork, device_ids=[0, 1])
        # 将模型转到GPU
    if opt.use_gpu:
        netWork.cuda()
        cudnn.benchmark = True
    # step2: 加载数据
    test_data = XueLangDataSet(opt.test_data_root, test=True)
    #batch_size=1 目前只支持一张图片预测   num_workers=1
    test_dataloader=DataLoader(test_data,batch_size=1,shuffle=False,num_workers=opt.num_workers)
    #存放预测结果
    results = []
    # 迭代数据集加载器
    for ii, (test_data_origin,test_img_name) in enumerate(test_dataloader):
        # test_input_img为模型输入图像的truple
        test_input_img=Variable(test_data_origin,volatile=True)

        if opt.use_gpu:
            test_input_img=test_input_img.cuda()


        # 测试集batch为1,压缩第0维
        test_input_img=t.squeeze(test_input_img,dim=0)
        test_label=netWork(test_input_img)
        # 概率  通过softmax可得概率 一张图得到多个结果  shape:[X,2]
        test_label_score = t.nn.functional.softmax(test_label,dim=1)
        #score_order即为有瑕疵得分，从大到小排序
        score_order,_=t.sort(test_label_score.data[:, 1],descending=True)
        # 取概率最大值作为最后结果
        score = score_order[0]
        if score==1:
            score=0.999999
        if score<0.000001:
            score=0.000001
        batch_results = [(test_img_name[0],score)]
        results = results + batch_results
    # 将测试结果写入csv文件中
    write_csv(results, opt.result_file)
    print("============测试完毕=============")



def write_csv(results,file_name):
    import csv
    #调整为写入模式
    with open(file_name,'w') as f:
        writer=csv.writer(f)
        # 写入标题
        writer.writerow(['filename','probability'])
        #写入元组数据
        writer.writerows(results)
if __name__ == '__main__':


    #====================训练部分=========================
    #生成用于计算IOU的txt
    xml_txt_IOU
    #生成数据集的txt
    xml_txt_dataset
    #开始训练
    train()


    # ====================测试部分=========================
    #开始测试
    # test()

