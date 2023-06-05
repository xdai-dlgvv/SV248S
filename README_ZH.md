
[**English**](./README.md) | [**中文**](./README_ZH.md)

# 单目标跟踪数据集 - SV248S [翻译中]
建议数据集中的数据来源于[吉林一号视频卫星](http://www.jl1.cn/EWeb/)。视频的分辨率约为0.92米。卫星捕获的视频帧率是 10 FPS，但我们在此数据集中使用的视频帧率是 25 FPS，这是因为开源的官方视频使用了视频帧插值 (VFI) 进行重新编码。为了尽可能避免 VFI 对目标像素造成不良影响，我们从 6 个视频中选择了 248 个受 VFI 影响较小的好目标。

数据集 (53.1GB) 是免费和开源的，但仅可用于非商业用途。该数据集的详细描述见“**[Deep Learning-Based Object Tracking in Satellite Videos: A Comprehensive Survey With a New Dataset](https://ieeexplore.ieee.org/document/9875020)**”。该数据集由西安电子科技大学[IPIU实验室](https://ipiu.xidian.edu.cn/)（*智能感知与图像理解教育部重点实验室*）于 2021 年 7 月完成。

该数据集还为小目标提供了蒙版级别的标注标准，这与直立边界框有很大不同。目标及其背景被一系列点组成的紧密多边形区分。该数据集中使用的 6 个视频具有不同的跟踪环境，您可以根据您的实验将它们拆分为训练/验证/测试集。

数据集文件按以下结构排列：
``` shell
- 01
|---- sequences
   |---- 000000
      |---- 000001.tiff
      |---- 000002.tiff
      |---- ...
   |---- 000001
   |---- ...
|---- annotations
   |---- 000000.abs
   |---- 000000.attr
   |---- 000000.poly
   |---- 000000.rect
   |---- 000000.state
   |---- ...
- 02
- ...

```

**文件定义**
- `tiff`: 序列的图像帧，这些帧被裁剪成相同大小，从 "000001"开始命名；
- `abs`: 一个 json 文件，它给出了源视频和目标信息的简短描述。 一个例子如下:
```json
{
    "source_info":{
        "video_id": "01",      # 源视频的名称
        "seq_id": "000002",    # 当前目标的名称
        "frame_range": [],     # 相对于原始视频使用的帧范围
        "crop_range": []       # 相对于原始视频的裁剪块的范围
    },
    "details":{
        "init_rect": [],       # 初始化边界框
        "init_poly": [],       # 初始化的紧密多边形
        "length": 325,         # 序列总长度
        "class_name": "ship",  # 类别名称：ship, plane, car, car-large  (这些名称与论文中不同)
        "level": "simple"      # 跟踪难度等级：simple, normal, hard
    }
}
```
***注意：论文中的类名与该 json 文件不同.*** `ship -> ship, airplane -> plane, vehicle -> car, large-vehicle -> car-large`

- `attr`: 序列属性文件，以csv格式保存；
- `state`: 帧标记文件，0 表示目标正常可见，1 表示目标不可见（背景簇），2 表示目标被遮挡；
- `rect`: 使用 (left_top_x, left_top_y, width, height) 直立边界框注释的目标；
- `poly`: 使用多个点的掩码级多边形注释的目标，例如：((x0,y0), (x1, y1), ...)。

## 248 个目标
建议数据集包含从 6 个不同的场景选择的 248 个移动目标。该数据集包括 4 个类别：车辆、大型车辆、船舶和飞机。其中车辆目标数量最多，涉及比其他类别更多的难样本。该数据集主要关注小目标，因为它们在卫星视频中被提出最多的跟踪困难。

## 3 个帧标记
帧标记记录在 `.state` 文件中，每个帧有一个整数表示被跟踪目标的状态：`NOR->0、INV->1、OCC->2`。
- **[INV]** *Invisiable* 目标在没有任何遮挡物的情况下消失，或者与周围环境太相似。
- **[NOR]** *Normal Visiable* 目标是可见的并且很容易找到。
- **[OCC]** *Occlusion* 目标在建筑物的阴影中或在某物体后面（例如桥梁和高楼）。

## 10个序列属性
序列属性以csv格式记录在`.attr`文件中（连续10个整数，以逗号分隔）。这些整数分别表示`STO,LTO,DS,IV,BCH,SM,ND,CO,BCL,IPR`。

这些属性有特殊的含义，列出如下：
- **[STO]** ***Short-Term Occlusion***: 序列存在小于等于 50 个连续帧的 OCC 标记。
- **[LTO]** ***Long-Term Occlusion***: 序列存在多于 50 个连续帧的 OCC 标记。
- **[DS]** ***Dense Similarity***: 被跟踪目标 2.5 倍的 OS 区域内存在一个或更多相似物体。
- **[IV]** ***Illumination Variation***: 被跟踪目标在亮度或颜色上存在可见变化。
- **[BCH]** ***Background Change***: 被跟踪目标的背景在颜色和纹理上存在可见变化。
- **[SM]** ***Slow Motion***: 被跟踪目标的移动速度小于 2.2 像素每秒。
- **[ND]** ***Natural Disturbance***: 被跟踪目标的外观受烟雾、沙尘天气的影响或被云层遮挡。
- **[CO]** ***Continuous Occlusion***: 序列中 2 次或多次发生 STO 或 LTO。
- **[BCL]** ***Background Cluster***: 序列至少 10 帧包含 INV 标记。
- **[IPR]** ***In-Plane Rotation***: 被跟踪目标具有大于等于 30 度的平面内旋转。

## 获取 SV248S
请将您的学校或单位名称发邮件给我们：liyuxuan_xidian@126.com，我们会给您一个私密的云盘链接。

## 引用
如果这项工作对您的研究有帮助，请引用：
```
@ARTICLE{9875020,
  author={Li, Yuxuan and Jiao, Licheng and Huang, Zhongjian and Zhang, Xin and Zhang, Ruohan and Song, Xue and Tian, Chenxi and Zhang, Zixiao and Liu, Fang and Yang, Shuyuan and Hou, Biao and Ma, Wenping and Liu, Xu and Li, Lingling},
  journal={IEEE Geoscience and Remote Sensing Magazine}, 
  title={Deep Learning-Based Object Tracking in Satellite Videos: A comprehensive survey with a new dataset}, 
  year={2022},
  volume={10},
  number={4},
  pages={181-212},
  doi={10.1109/MGRS.2022.3198643}}

```