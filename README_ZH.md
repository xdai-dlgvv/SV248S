
[**English**](./README_EN.md) | [**中文**](./README_CN.md)

# Single Object Tracking Dataset - SV248S [翻译中]
建议数据集中的数据来源于[吉林一号视频卫星](http://www.jl1.cn/EWeb/)。 视频的分辨率约为0.92米。 卫星捕获的视频是 10 FPS，但我们在此数据集中使用的视频是 25 FPS。 原因是这些开源的官方视频是用视频帧插值（VFI）重新编码的。 为了尽可能避免这种对目标像素的不良影响，我们从6个视频中选择了248个受VFI影响较小的好目标。

数据集 (53.1GB) 是免费和开源的，但仅可用于非商业用途。 该数据集的详细描述见“**[Deep Learning-Based Object Tracking in Satellite Videos: A Comprehensive Survey With a New Dataset](https://ieeexplore.ieee.org/document/9875020)**”。 该数据集由西安电子科技大学[IPIU实验室](https://ipiu.xidian.edu.cn/)（*智能感知与图像理解教育部重点实验室*）于2021年7月完成。

这个数据集还为小目标提供了蒙版级别的标注标准，这与旋转包围框有很大不同。 目标及其背景被一系列点的紧密多边形进行区分。 该数据集中使用的六个视频具有不同的跟踪环境，您可以根据您的实验将它们拆分为训练/验证/测试集。由于目标数量并不多，因此在本论文以及后续的延伸工作中，建议作为单目标跟踪的测试集。

这些文件按以下结构排列：
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

**FILE DIFINATION**
- tiff: 一个序列的帧图像，这些帧被裁剪成一个序列的相同大小，它们的名字从 "000001".
- abs: 一个 json 文件，它给出了源视频和目标信息的简短描述。 一个例子是由:
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
        "init_poly": [],       # 初始化的紧致多边形
        "length": 325,         # 序列的总长度
        "class_name": "ship",  # ship, plane, car, car-large  (这些名称与论文中不同)
        "level": "simple"      # simple, normal, hard
    }
}
```
***注意：论文中的类名与这个 json 文件不同.*** `ship -> ship, airplane -> plane, vehicle -> car, large-vehicle -> car-large`

- attr: 序列属性文件，保存为csv格式
- state: the frame flag file, 0 for normal visiable, 1 for invisiable (background cluster), 2 for occlusion
- rect: the annotated target which is represented by upright bounding boxes with (left_top_x, left_top_y, width, height).
- poly: the annotated target which is represented by mask-level polygon with multiple pairs of points, like: ((x1,y0), (x2, y2), ...)

## 248 Targets from Six Videos with Four Classes
There are 248 moving targets selected from six different scenarios. The proposed dataset includes 4 classes: Vehicle, Large-Vehicle, Ship and Airplane. Vehicles are the most targets, involving more hard samples than the others. This dataset is mainly focus on the small targets because they have raised the most tracking difficulties in Satellite Videos.

## Three Frame Flags
These flags are recorded in the `.state` Files with an integer number for each frame: NOR->0, INV->1, OCC->2.
- **[INV]** *Invisiable* The object is disappeared without any occluder or is too similar to its surroundings.
- **[NOR]** *Normal Visiable* The object is visiable and found easily.
- **[OCC]** *Occlusion* The object is in the shadow of the building or behind something (such as bridges and tall buildings).

## Ten Sequence Attributes
These attributes are recorded in the `.attr` Files with csv format (10 integer numbers in a row, splitted by comma). These numbers represent: `STO,LTO,DS,IV,BCH,SM,ND,CO,BCL,IPR`

These attributes have special meanings, listed below:
- **[STO]** ***Short-Term Occlusion***: The sequence exists less than or equal to 50 consecutive frames with OCC flags.
- **[LTO]** ***Long-Term Occlusion***: The sequence exists more than 50 consecutive frames with OCC flags.
- **[DS]** ***Dense Similarity***: One or more similar objects exist around the tracked object in the range of 2.5 times OS.
- **[IV]** ***Illumination Variation***: The object has noticeable changes in brightness or color.
- **[BCH]** ***Background Change***: The background of the tracked object has noticeable changes in color or texture.
- **[SM]** ***Slow Motion***: The moving speed of the tracked object is less than 2.2 pixels per second.
- **[ND]** ***Natural Disturbance***: The object’s appearance is influenced by smog or sandy weather or blocked by clouds.
- **[CO]** ***Continuous Occlusion***: STO or LTO occur twice or more times in a sequence.
- **[BCL]** ***Background Cluster***: There are at least ten frames that contain the INV flag.
- **[IPR]** ***In-Plane Rotation***: The object has an in-plane rotation at an angle greater than or equal to 30 degrees.

## Get SV248S
Please send E-mail to us with your school or origination name: liyuxuan_xidian@126.com, and we will give you a private link of Cloud Drive.

## Cite This
If this work is helpful for your research, please cite with:
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