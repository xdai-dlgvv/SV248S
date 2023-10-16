# Evaluation Tools
We provide two kinds of evaluation codes for this dataset.

- VOTtookit-based tracking/evaluation framework [Reconstructing in branch dev_tools]
- [Another third-part evaluation framework](https://github.com/Wprofessor/SV248S_toolkit/tree/main) [Enabled]

NOTE THAT: The third-part evaluation tool is based on the RECTANGLE BOUNDING BOX GROUNDTRUTH, so the ENUS score might be different from the score in original paper.

# Single Object Tracking Dataset - SV248S
The data in proposed dataset is derived from the [Jilin-1 Video Satellite](http://www.jl1.cn/EWeb/). The resolution of the video is about 0.92 meter. The video captured by the Satellite is 10 FPS however the video we used in this dataset is 25 FPS. The reason is that these open-souce official video are re-encoded with Video Frame Interpolation (VFI). In order to avoid this bad influence on the target pixels as much as possible, we selected 248 good targets which have less effected by VFI from 6 videos.

The dataset (53.1GB) is free and open-source but only availabe for non-commercial use. The detail description of this dataset is presented in "**[Deep Learning-Based Object Tracking in Satellite Videos: A Comprehensive Survey With a New Dataset](https://ieeexplore.ieee.org/document/9875020)**". This dataset was finished at July 2021 by [IPIU Lab](https://ipiu.xidian.edu.cn/) (*Key Laboratory of Intelligent Perception and Image Understanding of Ministry of Education*) of Xidian University.

This tentative dataset also provides a mask-level annotation criterion for small targets, which is quite different from the oriented BBox. The target and its background are splited by a tight-polygon containing a series of points. The six videos used in this dataset have different tracking environment, and you can split them into train/val/test set according to your experiment propose.

The files are arranged by the following structure:
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
- tiff: frame images for a sequences, these frames are cropped with the same size for a sequence, and their names start from "000001".
- abs: a json file that give a short description of the source video and target information. An example is given by:
```json
{
    "source_info":{
        "video_id": "01",      # the source video name
        "seq_id": "000002",    # current target sequence name
        "frame_range": [],     # the frame range used relative to original video
        "crop_range": []       # the cropped patch range relative to original video
    },
    "details":{
        "init_rect": [],       # the initialize bounding box
        "init_poly": [],       # the initialize tight-polygon
        "length": 325,         # the total length of this sequence
        "class_name": "ship",  # ship, plane, car, car-large  (these names are different with the survey)
        "level": "simple"      # simple, normal, hard
    }
}
```
***Notice: the class names in survey are different from this json file.*** `ship -> ship, airplane -> plane, vehicle -> car, large-vehicle -> car-large`

- attr: the sequence attribute file, saved in csv format.
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
