# 视觉计算辅助良品检测

- 参赛地址：[雪浪制造AI挑战赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=a2c22.11695015.1131732.1.4ea25275NNvZuf&raceId=231666) 


----------

# 版本


  - ## xuelang (排名：32/2403)
  
<div align="center">
<img src="http://boboprivate.oss-cn-beijing.aliyuncs.com/18-8-7/89764672.jpg" width="800px"  height="400px" alt="图片说明" >
</div>


  - ## XuelangYOLOv1ByBobo
  
  预训练模型：（仅适用有xml标注的数据，效果不好）

 ###### 注： [最新模型](https://pan.baidu.com/s/1wv_30IjungQO5fp5b0bKow)    [验证集最好的模型](https://pan.baidu.com/s/1mmgI78s5YUrMA9Y4RVM5yw) 

 - ## Resnet152

  支持网络模型：VGG16、VGG19、ResNet18、ResNet152（使用原图训练）
  
  结果：

| 模型（仅训练分类层） | 原图resize的大小 | 结果(线上AUC) |
|--------------------|------------------|---------------|
| resnet152          | 224              | 0.72          |
| vgg19              | 420              | 0.753         |

----------


  特别鸣谢：朱辉师兄
  




