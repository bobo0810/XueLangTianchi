# 视觉计算辅助良品检测

- 参赛地址：[雪浪制造AI挑战赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=a2c22.11695015.1131732.1.4ea25275NNvZuf&raceId=231666) 

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

 - 环境：

    | python版本  |  pytorch版本 |
    | ----------- | ----------   |
    |  3.5  | 0.3.0   |



 - 说明：

   1、运行main.py请确保启动可视化工具visdom
   2、所有参数均在config.py中设置即可

----------

# 版本：

  - ### YOLO v1版

（仅适用有xml标注的数据，效果不好）
  
  预训练模型：

  [最新保存的模型](https://pan.baidu.com/s/1wv_30IjungQO5fp5b0bKow) 

  [验证集最好的模型](https://pan.baidu.com/s/1mmgI78s5YUrMA9Y4RVM5yw) 

 - ### Resnet152

（使用原图训练，未进行裁剪）

  支持网络模型：VGG16、VGG19、ResNet18、ResNet152
  
  结果：（均为未训练完成即提交）

| 模型（仅训练分类层） | 原图resize的大小 | 结果(线上AUC) |
|--------------------|------------------|---------------|
| resnet152          | 224              | 0.72          |
| vgg19              | 420              | 0.753         |

----------
# 提升思路：

- 模型全局训练

- 数据增强

- 数据裁剪