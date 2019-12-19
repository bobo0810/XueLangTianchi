# 视觉计算辅助良品检测   

- 参赛地址：[雪浪制造AI挑战赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=a2c22.11695015.1131732.1.4ea25275NNvZuf&raceId=231666)  

- 排名：32/2403

----------

# 环境


| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.5        | 0.3.0       | Ubuntu |

----------

# 数据集


- 官方数据集结构

项目默认data文件夹下为已解压的数据集

```
data
│
└───xuelang_round1_train_part1_20180628
│   │   吊纬
│   │   毛斑
│   │   正常
│   │   ...
│   
└───xuelang_round1_train_part2_20180705
│   │   吊纬
│   │   毛斑
│   │   正常
│   │   ...
└───xuelang_round1_train_part3_20180709
│   │   吊纬
│   │   毛斑
│   │   正常
│   │   ...
└───xuelang_round1_test_a_20180709
│
└───xuelang_round1_test_b
```

######  注：[训练集下载](https://pan.baidu.com/s/1Y1MspixgeQtXECXvcds6OA)

- 训练使用的数据集

生成的xuelangtrainpart123_dataset.txt 为训练时用的数据集，训练过程中当加载每张图时才进行随机裁剪。

将官方数据集分为两部分：无瑕疵原图、有瑕疵原图。该数据集使用了一份无瑕疵原图、两份有瑕疵原图。

###### （注：一份无瑕疵原图用于产生 无瑕疵训练样本，一份有瑕疵原图用于产生  无瑕疵训练样本，一份有瑕疵原图用于产生 有瑕疵训练样本。）

![](https://github.com/bobo0810/imageRepo/blob/master/img/7134271.jpg)

从有瑕疵原图中产生有瑕疵训练样本分为四种情况：

随机从该图中的所有bbox中抽取一个bbox。

当抽取的bbox的w、h<420时从橘色方块部分随机产生裁剪样本左上角坐标，宽高均为420。

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/87211553.jpg" width="600px" height="400px" alt="图片说明" >
</div>


当抽取的bbox的w<420、h>420时从橘色方块部分随机产生裁剪样本左上角坐标，宽高均为420。

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/53243672.jpg" width="600px" height="400px" alt="图片说明" >
</div>



当抽取的bbox的w>420、h<420时从橘色方块部分随机产生裁剪样本左上角坐标，宽高均为420。

<div align="center">
<img src="https://github.com/bobo0810/imageRepo/blob/master/img/22474338.jpg" width="600px" height="400px" alt="图片说明" >
</div>

当抽取的bbox的w、h>420时 0.5概率使用原bbox，0.5概率在原bbox中随机裁剪420尺寸。

###### （注：详细代码在XueLangDataSet文件中）



----------

# 训练

- 1、开启Visdom
（Visdom类似TnsorFlow的tensorboard,可视化工具）
```
# First install Python server and client
pip install visdom
# Start the server 
python -m visdom.server
```
开始训练后浏览器访问 http://localhost:8097/ ,在Environment里选择XueLang_ResNet152_bo即可看到可视化界面。



- 2、准备验证集

下载[finaVal验证集](https://pan.baidu.com/s/13WEVsAvMp1fyQnZd6N15FA)，在code/utils/config.py中val_data_root属性配置好路径。

###### （注：该验证集与线上auc得分误差在[0.001-0.02]之间）

- 3、开始训练

执行main.py中的训练部分即可，通过vidom查看损失情况及在验证集上的AUC分数。

###### （注：默认多GPU训练，可在config文件中取消)

----------

# 测试

- 1、配置模型

在code/utils/config.py中给load_model_path属性配置 最终模型的路径。

######  [auc为0.929的模型下载](https://pan.baidu.com/s/1pmSAS85Gjc9b5SCXPMmppw)

- 2、配置测试集

在code/utils/config.py中给test_data_roots属性配置  测试集文件夹路径。

- 3、开始测试

执行mian.py中的测试部分即可，执行完成后，将生成csv并保存到文件夹submit/*.csv。

----------

# 思想

### 数据集准备

  - 动态裁剪： 训练过程中当开始加载每张图时才进行随机裁剪。

   ###### （注：裁剪仍过于严格，按照IOU放宽范围可试）


  - 数据挖掘：有瑕疵原图裁剪无瑕疵样本。

  - 多尺度融合：按照概率，0.5几率选择原图训练，0.5几率选择裁剪样本训练。

  ###### （注：前排大佬选择多模型融合，即原图训练模型，裁剪样本训练模型，最终结果融合。但比赛限制模型数量）

 #### 预测数据

 - 验证与预测时仅支持batch为1：一张图产生  420大小，半步长210滑动的96张裁剪样本 + 1张原图。
  

----------

# 说明

- 所有参数均在config.py中配置，十分方便。

- 提升方法：

   1、该项目未使用各种数据增强的方法，可试。

   2、该项目输入尺寸为224，可增大尺寸训练。

   3、该项目网络仅为单模型Resnet152，前排大佬使用多模型融合，可试。

   4、....

----------

# Q&A

Q：为何按照420裁剪，最后又resize到224训练？

A:  选择420是因为服务器最大可按照420训练，再大就out of memory。之后又选择resize到224训练是因为我后来发现 使用pytorch提供的Resnet152预训练模型，该模型为Imagenet按照224训练得到，故resize到224训练效果会更好。

Q：该比赛最值得说道的一点是什么？

A: 通过part123数据集精心抽取的原图作为测试集，使得该验证集与线上auc得分误差在[0.001-0.02]之间，且误差不随线上更换数据集而改变，故可以在本地无限次测试~



----------


  特别鸣谢：朱辉师兄



