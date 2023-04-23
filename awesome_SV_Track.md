# Awesome Satellite Video Tracking

本文档主要是对卫星视频，卫星视频跟踪的相关内容进行整理。以方便后续的研究者更快的入门。
本文档计划涉及以下任务：
一、单目标跟踪
二、多目标跟踪
三、运动目标检测

由于遥感领域的特殊性，大部分论文并不开源其跟踪代码。本文档第一大部分，主要对开源的代码进行整理，并根据情况对部分代码进行测试标记，以方便后续研究者参考。

由于我们测试标记也具有时效性与随机性（譬如，我们环境跟作者不同导致的性能差异，作者更新了链接提供了完整测试代码等），因此这项工作仅仅是尝试完整反应我们测试的结果。如果结果与你的结果不相符，欢迎联系我们一起对这些内容进行探讨。毕竟卫星视频的数据的测试结果很有可能跟参数的关系特别大。这个需要大家自行甄别。

## 开源的代码

模板：

1. **方法简写**: 论文全称 [[Paper]]() [[Code]]() 
发表刊物：刊名，年份

### SOT

1. **CFME**: Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations 
[[Paper]]() [[Code]](https://github.com/SY-Xuan/CFME) 
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2020

1. **CoCRF-TrackNet**: A Collaborative Learning Tracking Network for Remote Sensing Videos 
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9819825) [[Code]](https://github.com/Dawn5786/CoCRF-TrackNet) 
发表刊物：IEEE Transactions on Cybernetics， 2022
未公开权重。

1. **ThickSiam**: High-resolution Satellite Video Object Tracking Based on ThickSiam Framework 
[[Paper]]() [[Code]](https://github.com/CVEO/ThickSiam) 
发表刊物：GIScience \& Remote Sensing, 2023
提供了论文使用的数据集，没有提供测试代码。

1. **MACF**： 
[[Paper]] No paper [[Code]](https://github.com/binlin-cv/MACF)
发表刊物：
获得了ICPR 2022运动目标检测和单目标检测。代码未上传。

### MOT

1. **Adaptive Birth for the GLMB Filter for object tracking in satellite videos** [[Paper]](https://ieeexplore.ieee.org/abstract/document/9943411/) [[Code]](https://github.com/binlin-cv/MACF)
发表刊物：2022 IEEE 32st International Workshop on Machine Learning for Signal Processing (MLSP)

1. **TGraM**: Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling [[Paper]]() [[Code]](https://github.com/zuzi2015/TGraM)
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2022

### 数据集
1. **SV248S**: Deep Learning-Based Object Tracking in Satellite Videos: A Comprehensive Survey With a New Dataset [[paper]](https://ieeexplore.ieee.org/document/9875020) [[code]](https://github.com/xdai-dlgvv/sv_dataset)
发表刊物：IEEE Geoscience and Remote Sensing Magazine， 2022

3. **VISO**: Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark 
[[Paper]]() [[Code]](https://github.com/QingyongHu/VISO) 
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2021

1. **ThickSiam**: High-resolution Satellite Video Object Tracking Based on ThickSiam Framework 
[[Paper]]() [[Code]](https://github.com/CVEO/ThickSiam) 
发表刊物：GIScience \& Remote Sensing, 2023

1. **TGraM**: Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling 
[[Paper]]() [[Code]](https://github.com/zuzi2015/TGraM)
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2022

## 相关论文
### SOT
### MOT
### 运动目标检测