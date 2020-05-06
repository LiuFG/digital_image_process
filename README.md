# digital_image_process

数字图像处理学习笔记, 根据以下学习指南进行:

1. 学习用书： 数字图像处理 MATLAB 版 冈萨雷斯著

   > 推荐两本书,一本偏理论, 一本偏代码实战(Matlab), 打包下载地址: 链接：https://pan.baidu.com/s/1Ir3vW_GwTOXbX-QIHkLkxA 
   > 提取码：nd2a 
   > 
   >
   > 偏理论: \<数字图像处理\_第三版\_冈萨雷斯\>
   >
   > 偏实战: \<数字图像处理(MATLAB版)(中译版)[冈萨雷斯]\>
   >
   > 包含书里的图片素材

2. 学习内容： 第 1 章、 第 2 章、 第 3 章、 第 6 章、 第 9 章的 9.1\~9.3 小节、
   第 10 章的 10.1\~10.3 小节、 第 11 章的 11.1~11.2 小节

3. 学习要求：为了增强理解，需要根据自己对方法的理解编写相应的 matlab程 序，
   书中的示例程序可以作为参考。

4. 图像处理高阶算法
   学 习 内 容 ： SIFT[1], LBP[2], BoW[3], affine transformation[4], similarity
   transformation[5]
   
   [1] material: https://blog.csdn.net/zddmail/article/details/7521424
   
   code: https://github.com/sun11/sw-sift (matlab 版)
   
   https://github.com/paulaner/SIFT (python 版)
   
   [2] material: https://blog.csdn.net/heli200482128/article/details/79204008
   
   code: https://github.com/michael92ht/LBP （ python 版）
   
   https://github.com/bikz05/texture-matching （ python 版）
   
   [3] material: https://blog.csdn.net/tiandijun/article/details/51143765
   
   code: https://github.com/qijiezhao/SIFT_BOW_Usage （ python 版）
   
   https://github.com/lucifer726/bag-of-words- （ python 版）
   
   [4] material: https://www.cnblogs.com/ghj1976/p/5199086.html
   
   [5] material: https://blog.csdn.net/u014096352/article/details/53526747

## 说明 

由于MATLAB基本操作之前学过，主要对遗忘的内容进行笔记;    
除了MATLAB代码，加入相关Python3代码。其中使用[opencv-python](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html)，
[pillow（PIL）](https://pillow.readthedocs.io/en/latest/handbook/index.html)，
[skimage](https://scikit-image.org/docs/dev/user_guide)等工具。     

> 很久之前的笔记, 适合CV初学者进行学习

## 内容列表（已完成）     

* CH01_CH02:MATLAB数据类型与基本操作;       
* CH03:gamma亮度与对比度拉伸变换，直方图均衡化与匹配，空间滤波
* CH06:索引图像概念，图像提取&分离通道，彩色空间转换，彩色映射变换，彩色图像空间滤波，彩色边缘检测，彩色图像分割
* CH09:膨胀，腐蚀，顶帽，黑帽，击中或击不中变换，查找表   
* CH10:点、线检测(canny、Hough)，阈值处理   
* CH11:链码、最小周长多边形的多边形近似、标记、边界片段、骨骼
* 高阶:SIFT
