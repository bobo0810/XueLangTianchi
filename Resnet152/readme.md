# 视觉计算辅助良品检测

- 参赛地址：[雪浪制造AI挑战赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=a2c22.11695015.1131732.1.4ea25275NNvZuf&raceId=231666) 


----------

 - 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.3.0   |



 - 说明：

   1、运行main.py请确保启动可视化工具visdom

   2、所有参数均在config.py中设置即可

----------


 - ### Resnet152

（使用原图训练，未进行裁剪）

  支持网络模型：VGG16、VGG19、ResNet18、ResNet152
  
  结果：（均为未训练完成即提交）

| 模型（仅训练分类层） | 原图resize的大小 | 结果(线上AUC) |
|--------------------|------------------|---------------|
| resnet152          | 224              | 0.72          |
| vgg19              | 420              | 0.753         |

----------

# 准备数据
  
  - train_resnet152_paert123.txt内容格式为

| 数据集地址 | 说明 |标签 |
|------------|------|------|
| .../origin/..  | origin为"正常"文件夹  | 0(无瑕疵) |
| ../乱码/..      | 中文乱码为其余文件夹  |1(有瑕疵)  |

- 下载官方数据集，将train_resnet152_paert123.txt中的路径前缀替换为本机路径即可。

- 在config.py中配置data_root路径为本机train_resnet152_paert123.txt的路径。

# 训练

 #### 注：所有参数均在config.py中进行配置

1、开启Visdom（类似TnsorFlow的tensorboard,可视化工具）
```
# First install Python server and client
pip install visdom
# Start the server 
python -m visdom.server
```
2、选择网络

在config.py中将所选网络取消注释即可。

目前支持VGG16\VGG19\ResNet18\ResNet152

3、运行main.py中的train()即可。

通过visdom查看loss等的可视化。

# 测试

1、下载官方测试集，在config.py中test_data_root配置测试集地址。

2、在config.py中load_model_path配置训练好的模型地址。

3、运行main.py中的test()即可。



----------

# 项目结构

- 总结构

  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/99053959.jpg)
  
  
- 一般项目结构

  1、定义网络
  
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/16409622.jpg) 
  
   2、封装数据集
   
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/38894621.jpg)
  
   3、工具类
   
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/98583532.jpg)
  
   4、主函数
   
  ![](http://boboprivate.oss-cn-beijing.aliyuncs.com/18-5-26/32257225.jpg)
  
  
----------

# 提升思路：

- 模型全局训练

- 数据增强

- 数据裁剪