# Awesome Satellite Video Tracking

本文档主要是对卫星视频、卫星视频跟踪的相关内容进行整理，以方便后续的研究者更快的入门。
本文档计划涉及以下任务：  
一、单目标跟踪  
二、多目标跟踪  
三、运动目标检测 

由于遥感领域的特殊性，大部分论文并不开源其代码。本文档第一部分，主要对开源的代码进行整理，并根据情况对部分代码进行测试标记，以方便后续研究者参考。

由于我们测试标记也具有时效性与随机性（譬如，我们环境跟作者不同导致的性能差异，作者更新了链接提供了完整测试代码等），因此这项工作仅仅是尝试完整反映我们测试的结果。如果这与你的结果不相符，欢迎联系我们一起对这些内容进行探讨。毕竟卫星视频数据的测试结果很有可能跟参数的关系特别大。这个需要大家自行甄别。

## 开源的代码

<!-- 模板：

1. **方法简写**: 论文全称 [[Paper]]() [[Code]]() 
发表刊物：刊名，年份  
摘要： -->

### SOT

1. **CFME**: Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/8880656) [[Code]](https://github.com/SY-Xuan/CFME)  
  发表刊物：IEEE Transactions on Geoscience and Remote Sensing, 2020   
  <!-- 摘要：作为一种新的地球观测方法，视频卫星能够通过提供高时间分辨率的遥感图像，连续监测地球表面上的特定事件。视频观测使得各种新的卫星应用成为可能，例如目标追踪和道路交通监测。本文针对卫星视频中快速目标追踪的问题，提出了一种基于相关滤波器和运动估计的新型追踪算法。基于核相关滤波器（KCF），该算法提供了以下改进：1）通过将卡尔曼滤波器和运动轨迹平均结合，提出了一种新颖的运动估计算法，以减轻KCF的边界效应；2）解决了当移动物体部分或完全被遮挡时追踪失败的问题。实验结果表明，我们的算法能够在卫星视频中以95%的准确率追踪移动物体。 -->

2. **CoCRF-TrackNet**: A Collaborative Learning Tracking Network for Remote Sensing Videos  
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9819825) [[Code]](https://github.com/Dawn5786/CoCRF-TrackNet)  
发表刊物：IEEE Transactions on Cybernetics, 2022
未公开权重。  
<!-- 摘要：随着遥感视频的日益易得，遥感跟踪逐渐成为一个热门问题。然而，在复杂的遥感场景中进行准确的检测和跟踪仍然是一个挑战。本文提出了一种用于遥感视频的协作学习跟踪网络，包括一致感受野并行融合模块（CRFPF）、双分支空间-通道协同注意力（DSCA）模块和几何约束重追踪策略（GCRT）。考虑到遥感场景中小尺寸物体很难被普通前向网络提取出有效特征，我们提出了CRFPF模块，建立具有一致感受野的并行分支，从浅层到深层分别提取特征，并灵活地融合层次化特征。由于难以区分目标和背景，所提出的DSCA模块使用空间-通道协同注意力机制协同学习相关信息，增强目标的显著性，并回归到精确的边界框。考虑到类似物体的干扰，我们设计了GCRT策略，通过估计的运动轨迹判断是否存在虚假检测，然后通过削弱干扰的特征响应来恢复正确的目标。多个数据集上的实验结果和理论分析证明了我们提出的方法的可行性和有效性。代码和网络可在https://github.com/Dawn5786/CoCRF-TrackNet 获取。 -->

3. **ThickSiam**: High-resolution Satellite Video Object Tracking Based on ThickSiam Framework  
[[Paper]](https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2163063) [[Code]](https://github.com/CVEO/ThickSiam)  
发表刊物：GIScience \& Remote Sensing, 2023  
提供了论文使用的数据集，没有提供测试代码。  
<!-- 摘要：高分辨率卫星视频实现了对地面指定区域的短期注视观察，使遥感数据的时间分辨率达到秒级别。卫星视频中的单目标跟踪（SOT）任务引起了相当大的关注。然而，这面临着复杂背景、目标特征表示不足以及缺乏公开可用的数据集等挑战。为了应对这些挑战，本工作设计了一个由厚化残差块孪生网络（TRBS-Net）提取鲁棒的语义特征以获得初始跟踪结果，并且由改进的卡尔曼滤波器（RKF）模块同时校正目标的轨迹和大小的ThickSiam框架。TRBS-Net和RKF模块的结果通过N帧收敛机制进行组合，以实现准确的跟踪。我们在我们的注释数据集上进行了消融实验，评估了所提出的ThickSiam框架和其他19个最先进的跟踪器的性能。比较结果表明，我们的ThickSiam跟踪器在一块NVIDIA GTX1070Ti GPU上以56.849 FPS的速度运行时，获得了0.991的准确度和0.755的成功度。 -->

4. **MACF**：  
[[Paper]]() No paper [[Code]](https://github.com/binlin-cv/MACF)  
发表刊物：  
获得了ICPR 2022运动目标检测和单目标检测。代码未上传。


### MOT

1. Adaptive Birth for the GLMB Filter for object tracking in satellite videos  
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9943411/) [[Code]](https://github.com/binlin-cv/MACF)  
发表刊物：2022 IEEE 32st International Workshop on Machine Learning for Signal Processing (MLSP)  
<!-- 摘要：广义标记多伯努利（GLMB）滤波器在多目标跟踪（MOT）中取得了显著的成果。然而，GLMB滤波器依赖于强假设，如对目标初始状态的先验知识。实际情况下，比如卫星视频目标跟踪，这些假设面临挑战，因为目标出现在随机位置，目标检测器会输出大量的误报。我们提出了GLMB滤波器的增强版，该滤波器通过学习之前的轨迹来估计准确的假设初始化。我们追踪之前的目标状态，并利用这些信息对新出现的目标的初始速度进行采样。这一添加显著提高了低帧率（FPS）视频中的GLMB性能，其中目标的初始状态对目标跟踪至关重要。我们测试了这个增强GLMB滤波器与可比较的跟踪器和以前的GLMB滤波器解决方案，结果表明我们的滤波器获得了更好的性能。代码可在https://github.com/Ayana-Inria/GLMB-adaptive-birth-satellite-videos 获取。 -->

1. **TGraM**: Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling  
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9715124) [[Code]](https://github.com/zuzi2015/TGraM)  
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2022  
<!-- 摘要：最近，卫星视频已成为地球观测的新兴手段，提供了追踪移动物体的可能性。然而，现有的多目标追踪器通常设计用于自然场景，未考虑遥感数据的特点。此外，大多数追踪器由检测和重新识别（ReID）两个独立阶段组成，这意味着它们无法相互促进。为此，我们提出了一种端到端的在线框架，称为TGraM，用于卫星视频中的多目标追踪。它从多任务学习的角度将多目标追踪建模为图信息推理过程。具体而言，提出了一种基于图的时空推理模块，用于挖掘视频帧之间的潜在高阶相关性。此外，考虑到检测和ReID之间优化目标的不一致性，设计了一种多任务梯度对抗学习策略，用于规范每个任务特定网络。此外，为了解决这一领域的数据稀缺性问题，建立了一个大规模高分辨率的吉林一号卫星视频多目标追踪数据集（AIR-MOT）进行实验。与最先进的多目标追踪器相比，TGraM实现了检测和ReID之间的高效协作学习，将追踪精度提高了1.2倍。代码和数据集将在网上提供（https://github.com/HeQibin/TGraM）。 -->

### 数据集
1. **SV248S**: Deep Learning-Based Object Tracking in Satellite Videos: A Comprehensive Survey With a New Dataset  
[[paper]](https://ieeexplore.ieee.org/document/9875020) [[code]](https://github.com/xdai-dlgvv/sv_dataset)  
发表刊物：IEEE Geoscience and Remote Sensing Magazine， 2022  
<!-- 摘要：作为卫星视频（SVs）研究中的一项基础任务，目标跟踪被用于交通评估、军事安全等领域中对感兴趣目标的跟踪。当前遥感领域中的卫星技术使得以相对较高的帧率和图像分辨率对移动目标进行跟踪成为可能。然而，这种特殊视角下的目标往往很小且模糊，使得有效提取深度特征变得困难。因此，在卫星视频中，提出了不少基于深度学习（DL）的目标跟踪方法。此外，用于日常生活视频（DLVs）的评估标准并不完全适用于卫星视频，这经常导致小物体的低精度评估结果。本文在卫星视频研究方面做出了三个贡献。首先，提出了一个新的单目标跟踪（SOT）数据集SV248S，其中包括248个序列，并进行了高精度手动注释，设计了10种属性标签来完全表示跟踪过程中的困难。其次，提出了两种针对小目标跟踪的高精度评估方法。最后，对2017年至2021年间流行的28种基于DL的最先进（SOTA）跟踪方法在提出的数据集上进行了评估和比较。此外，基于全面的实验结果，总结了有效采用基于DL的方法的一些建议。 -->

1. **VISO**: Detecting and Tracking Small and Dense Moving Objects in Satellite Videos: A Benchmark  
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9625976) [[Code]](https://github.com/QingyongHu/VISO)  
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2021  
<!-- 摘要：卫星视频能够对大范围区域进行连续观测，这对许多遥感应用非常重要。然而，在卫星视频中实现移动目标检测和跟踪仍然具有挑战性，原因是目标的外观信息不足，并且缺乏高质量的数据集。在本文中，我们首先建立了一个大规模卫星视频数据集，为移动目标检测和跟踪任务提供丰富的注释。该数据集由吉林一号卫星收集，由47个高质量视频组成，其中包含1,646,038个目标检测感兴趣实例和3,711个目标跟踪轨迹。然后，我们介绍了一种基于累积多帧差分和鲁棒矩阵补全的运动建模基线，以提高检测率并减少误报。最后，我们建立了第一个公开的卫星视频移动目标检测和跟踪基准，并在我们的数据集上广泛评估了几种代表性方法的性能。同时，提供了全面的实验分析和深入的结论。该数据集可在https://github.com/QingyongHu/VISO 获取。 -->

1. **ThickSiam**: High-resolution Satellite Video Object Tracking Based on ThickSiam Framework  
[[Paper]](https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2163063) [[Code]](https://github.com/CVEO/ThickSiam)  
发表刊物：GIScience \& Remote Sensing, 2023  
<!-- 摘要：高分辨率卫星视频实现了对地面指定区域的短期注视观察，使遥感数据的时间分辨率达到秒级别。卫星视频中的单目标跟踪（SOT）任务引起了相当大的关注。然而，这面临着复杂背景、目标特征表示不足以及缺乏公开可用的数据集等挑战。为了应对这些挑战，本工作设计了一个由厚化残差块孪生网络（TRBS-Net）提取鲁棒的语义特征以获得初始跟踪结果，并且由改进的卡尔曼滤波器（RKF）模块同时校正目标的轨迹和大小的ThickSiam框架。TRBS-Net和RKF模块的结果通过N帧收敛机制进行组合，以实现准确的跟踪。我们在我们的注释数据集上进行了消融实验，评估了所提出的ThickSiam框架和其他19个最先进的跟踪器的性能。比较结果表明，我们的ThickSiam跟踪器在一块NVIDIA GTX1070Ti GPU上以56.849 FPS的速度运行时，获得了0.991的准确度和0.755的成功度。 -->

1. **TGraM**: Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling  
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9715124) [[Code]](https://github.com/zuzi2015/TGraM)  
发表刊物：IEEE Transactions on Geoscience and Remote Sensing， 2022  
<!-- 摘要：最近，卫星视频已成为地球观测的新兴手段，提供了追踪移动物体的可能性。然而，现有的多目标追踪器通常设计用于自然场景，未考虑遥感数据的特点。此外，大多数追踪器由检测和重新识别（ReID）两个独立阶段组成，这意味着它们无法相互促进。为此，我们提出了一种端到端的在线框架，称为TGraM，用于卫星视频中的多目标追踪。它从多任务学习的角度将多目标追踪建模为图信息推理过程。具体而言，提出了一种基于图的时空推理模块，用于挖掘视频帧之间的潜在高阶相关性。此外，考虑到检测和ReID之间优化目标的不一致性，设计了一种多任务梯度对抗学习策略，用于规范每个任务特定网络。此外，为了解决这一领域的数据稀缺性问题，建立了一个大规模高分辨率的吉林一号卫星视频多目标追踪数据集（AIR-MOT）进行实验。与最先进的多目标追踪器相比，TGraM实现了检测和ReID之间的高效协作学习，将追踪精度提高了1.2倍。代码和数据集将在网上提供（https://github.com/HeQibin/TGraM）。 -->

1. **SatSOT**: A Benchmark Dataset for Satellite Video Single Object Tracking  
  [[Paper]](https://ieeexplore.ieee.org/document/9672083) [[Code]]() No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2022  
  <!-- 摘要：通过对特定区域进行连续成像，卫星视频在监视和交通管理等各种应用中展现出卓越的能力。尽管目标跟踪在近年来取得了重大进展，但由于缺乏开放源代码的卫星数据集，卫星目标跟踪的发展受到了限制。因此，建立一个卫星视频目标跟踪基准是填补这一空白并推进研究的重要任务。在这项工作中，我们提出了SatSOT，第一个密集标注的卫星视频单目标跟踪基准数据集。SatSOT包括105个序列，共27664帧，涵盖了卫星视频中四类典型移动目标：汽车、飞机、船和火车，并提供了11个属性。基于所提出的数据集以及卫星视频目标跟踪中存在的重要挑战，如小目标、背景干扰和严重遮挡，我们对15种最好和最具代表性的跟踪算法进行了详细评估和分析，为进一步研究卫星视频目标跟踪提供了基础。 -->

## 相关论文
### SOT

1. A Quantum Evolutionary Learning Tracker for Video  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10092876)  No code  
  发表刊物: IEEE Transactions on Evolutionary Computation , 2023   
  摘要：视频目标跟踪是计算机视觉领域的一个热门研究方向。随着视频数据的不断更新，更多特殊视角和具有挑战性的视频数据不断涌现。这给目标跟踪任务带来了挑战，并对模型的泛化能力提出了更高的要求。本文提出了一种新颖的量子进化学习视频跟踪器。该模型将量子进化与深度网络相结合，用于跟踪视频中的目标。模型利用量子进化学习跟踪器生成可靠的候选区域种群，并利用深度网络进行分类。特别地，量子进化预测器通过旋转算子和轨迹预测目标的运动状态，并为跟踪器提供运动状态信息。预测器可以整合目标的历史上下文信息，在外观特征失效的情况下为模型提供稳定的候选估计种群。量子进化和深度网络相结合形成了端到端的在线视频目标跟踪器。此外，我们提出了一种新的视频目标跟踪评估算法：平衡交并比。该评估算法利用长宽比平衡重叠和距离的比例。最后，我们在OTB 2015数据集上进行了自然视频的测试，并在SV248A10-SOT数据集上进行了卫星视频的测试。通过与二十多种经典跟踪器模型进行比较，分析和验证了所提出模型的性能。实验结果表明，我们的模型具有很高的泛化能力和鲁棒性。
  
1. **SiamMDM**: An Adaptive Fusion Network With Dynamic Template for Real-Time Satellite Video Single Object Tracking  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/10113336)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2023  
  摘要：最近，卫星视频中的移动目标跟踪引起了广泛关注。然而，卫星视频中目标跟踪的发展相对于一般视频而言进展缓慢，主要有以下关键原因。首先，卫星视频中的典型移动目标由很少的像素组成，失去了大部分的外观特征，使得跟踪器难以将目标与背景区分开来。其次，卫星视频目标的外观经常因为遮挡、光照变化或其他因素而发生变化。经典的Siamese跟踪网络仅使用第一帧作为目标模板，导致跟踪结果较差。第三，当目标完全遮挡时，跟踪器难以重新捕捉目标。为了解决上述问题，本文提出了一种基于多响应图融合和时空约束的Siamese跟踪网络。通过在跟踪网络的不同层生成响应图并进行自适应融合，可以更准确地跟踪卫星视频中的小目标。此外，还提出了一种动态模板更新策略，以应对卫星视频中目标外观的可能变化，减少对初始帧的高度依赖。为了重新捕捉目标，提出了一种基于得分引导的目标运动轨迹预测模型。我们将提出的Siamese跟踪网络简称为SiamMDM。我们在SatSOT和SV248S两个大型卫星视频目标跟踪数据集上进行了全面的实验。实验结果表明，我们的方法在每秒超过110帧（FPS）的速度下实现了最先进的跟踪性能。

1. Object Tracking Based on Satellite Videos: A Literature Review  
  [[Paper]](https://www.mdpi.com/2072-4292/14/15/3674)  No code  
  发表刊物: Remote Sens, 2022  
  摘要：近年来，视频卫星已成为地球观测的一种吸引人的方法，为连续监测特定事件提供了地球表面的连续图像。机载光学和通信系统的发展使得卫星图像序列的各种应用成为可能。然而，基于卫星视频的目标跟踪是遥感领域中一个具有挑战性的研究课题，因为其空间和时间分辨率相对较低。本调查系统地研究了当前基于卫星视频的目标跟踪方法和基准数据集，重点关注五种典型的跟踪应用：交通目标跟踪、船舶跟踪、台风跟踪、火灾跟踪和冰川运动跟踪。对每个跟踪目标的基本方面进行了总结，包括跟踪架构、基本特征、主要动机和贡献。此外，还讨论了流行的视觉跟踪基准及其各自的特性。最后，基于WPAFB视频生成了一个经过修订的多级数据集，并进行了定量评估，以促进卫星视频目标跟踪领域的未来发展。此外，选择了Difficulty Score (DS)较低的54.3%轨迹并将其命名为Easy组，而27.2%和18.5%的轨迹分别分组为Medium-DS组和Hard-DS组。
  
1. Object Tracking in Satellite Videos Based on Improved Kernel Correlation Filter Assisted by Road Information  
  [[Paper]](https://www.mdpi.com/2072-4292/14/17/4215)  No code  
  发表刊物: Remote Sens, 2022  
  摘要：卫星视频可以对地球表面的目标区域进行持续观测，获得高时间分辨率的遥感视频，从而实现对卫星视频中的物体进行跟踪。然而，卫星视频中的物体尺寸通常较小且缺乏纹理特征，同时卫星视频中的移动物体容易被遮挡，这对跟踪器提出了更高的要求。为了解决上述问题，考虑到遥感图像中包含丰富的道路信息，可以用来约束卫星视频中物体的轨迹，本文提出了一种改进的基于道路信息辅助的核相关滤波器（KCF）用于跟踪小物体，尤其是当物体被遮挡时。具体而言，本文的贡献如下：首先，重构了跟踪置信度模块，整合了响应图的峰值响应和平均峰值相关能量，以更准确地判断物体是否被遮挡。然后，设计了自适应卡尔曼滤波器，即根据物体的运动状态自适应调整卡尔曼滤波器的参数，提高了跟踪的鲁棒性，减少了物体遮挡后的跟踪漂移。最后，提出基于道路信息辅助的物体跟踪策略，利用道路信息作为约束来更准确地定位物体。经过上述改进，与KCF跟踪器相比，我们的方法在跟踪速度为每秒300帧的情况下，跟踪精度提高了35.9%，跟踪成功率提高了18.1%，满足实时要求。
  
1. Vehicle Tracking on Satellite Video Based on Historical Model  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9847077)  No code  
  发表刊物: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2022  
  摘要：由于特征较少、物体遮挡和相似物体外观，卫星视频中的车辆跟踪对现有的物体跟踪算法构成了挑战。为了提高物体跟踪算法的性能，本研究提出了一种专为卫星视频设计的基于历史模型的跟踪器。该跟踪器使用视频中每帧的历史模型来更新跟踪器。历史模型包含丰富的物体信息和背景信息，从而提高了对特征较少的物体的跟踪能力。此外，设计了一种历史模型评估方案，以获取可靠的历史模型，确保跟踪器对当前帧中的物体敏感，从而避免因物体外观和背景变化而产生的影响。为了解决由物体遮挡和相似物体外观引起的跟踪器漂移问题，还提出了一种防漂移跟踪器修正方案。通过在卫星视频数据集SatSOT上进行的对比实验，我们的跟踪器展现出了出色的性能。此外，进行了敏感性分析、不同标准的对比实验和消融实验，以证明所提出的方案在提高跟踪器的准确性和成功率方面是有效的。
  
1. Object Tracking in Satellite Videos: A Spatial-Temporal Regularized Correlation Filter Tracking Method With Interacting Multiple Model  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9786761)  No code  
  发表刊物: IEEE Geoscience and Remote Sensing Letters, 2022  
  摘要：卫星视频中的目标遮挡是常见的，这使得物体跟踪变得困难，因为大多数最先进的跟踪器对遮挡，特别是完全遮挡，不够稳健。在本文中，我们提出了一种新颖的相关滤波器算法，结合了相关滤波器和交互多模型（IMM）的优势，用于卫星视频中的目标跟踪。当目标被遮挡时，我们利用IMM来预测目标位置。因此，所提出的跟踪器对遮挡具有鲁棒性。实验结果表明，与最先进的方法相比，我们的跟踪器在目标遮挡时表现出色，并取得了优异的性能。
  
1. Object Tracking in Satellite Videos: Correlation Particle Filter Tracking Method With Motion Estimation by Kalman Filter  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9875357)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2022  
  摘要：卫星视频中的目标跟踪面临着目标遮挡、目标旋转和背景杂波等各种挑战。本研究提出了一种带有运动估计（ME）的相关粒子滤波器（CPF）算法，用于卫星视频中的目标跟踪。该跟踪器称为相关粒子卡尔曼滤波器（CPKF），结合了相关、粒子和卡尔曼滤波器的优势。与基于相关滤波器的现有跟踪方法相比，所提出的跟踪器具有三个主要优势：1）粒子采样和运动估计使其对部分和完全遮挡具有鲁棒性；2）颜色直方图模型使其对目标旋转具有鲁棒性；3）多个特征响应图的融合有效处理背景杂波和低对比度。实验结果表明，所提出的跟踪算法优于现有的方法。
  
1. Object Tracking in Satellite Videos Based on Siamese Network With Multidimensional Information-Aware and Temporal Motion Compensation  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9908539)  No code  
  发表刊物: IEEE Geoscience and Remote Sensing Letters, 2022  
  摘要：众多商业卫星的可用为遥感序列中的典型目标跟踪创造了有利条件，使其在众多应用中具有实用性。然而，在这一领域中，小目标、多个相似干扰物、背景杂波和遮挡等问题是重要的挑战。本研究提出了一种新颖的远程感知跟踪方法，即追踪器-时间运动补偿Siamese网络（Siam-TMC）。我们的方法依赖于一个多维信息感知（Dim-Aware）模块和一个时间运动补偿（TMComp）机制。值得注意的是，我们提出了一个基于双分支的Dim-Aware模块，将前景和高频信息结合起来，以区分重要的小目标和干扰物。此外，我们设计了一个使用时间运动信息的TMComp机制，通过遮挡检测的监督来减轻目标轨迹漂移。对基准数据集进行的详细实验表明，我们的方法在遮挡场景中表现优于现有的跟踪模型。

1. **HRSiam**: High-Resolution Siamese Network, Towards Space-Borne Satellite Video Tracking  
  [[Paper]](https://ieeexplore.ieee.org/document/9350236)  No code  
  发表刊物: IEEE Transactions on Image Processing, 2021  
  摘要：从太空卫星视频中跟踪移动物体是一项新的具有挑战性的任务。主要困难源于感兴趣目标的极小尺寸。首先，由于目标通常只占据少数像素，很难获得具有区分性的外观特征。其次，小物体容易遭受遮挡和光照变化，使得物体特征与周围区域的特征难以区分。目前的先进跟踪方法主要考虑低空间分辨率的单帧高级深度特征，并且很难从视频中获得帧间运动信息。因此，它们无法准确定位这种小物体并处理卫星视频中的具有挑战性的场景。在本文中，我们成功设计了一个轻量级的、具有高空间分辨率的并行网络，用于定位卫星视频中的小物体。这种架构保证了在应用于Siamese跟踪器时的实时和精确定位。此外，还提出了基于在线移动目标检测和自适应融合的像素级精化模型，以增强卫星视频中的跟踪鲁棒性。它实时对视频序列进行建模，以像素为单位检测运动目标，具有充分利用跟踪和检测的能力。我们在真实的卫星视频数据集上进行了定量实验，结果表明所提出的高分辨率Siamese网络（HRSiam）在超过30帧/秒的速度下实现了最先进的跟踪性能。
  
1.  **MBLT**: Learning Motion and Background for Vehicle Tracking in Satellite Videos  
  [[Paper]](https://ieeexplore.ieee.org/abstract/document/9533178)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2021  
  摘要：最近，卫星视频提供了一种动态监测地球表面的新方式。对卫星视频的解读越来越受到关注。在本文中，我们关注卫星视频中的车辆跟踪问题。卫星视频通常具有较低的分辨率，并导致以下问题：1）车辆目标的尺寸通常只包含少数像素；2）车辆通常具有相似的外观，很容易导致在观察区域内错误跟踪。常用的跟踪方法通常关注目标的表示并将其与背景区分开，这在以上问题上存在限制。因此，在本文中，我们提出通过学习目标的运动和背景来帮助跟踪器更准确地识别目标。我们提出了一个预测网络，基于全卷积网络（FCN）从先前的结果中学习，在每个像素中预测目标在下一帧中的位置概率。此外，引入了一种分割方法，用于生成每帧中目标的可行区域并为该区域分配高概率。为了定量比较，我们从吉林一号拍摄的九个卫星视频中手动注释了20个具有代表性的车辆目标。此外，我们还选择了两个公开卫星视频数据集进行实验。大量的实验结果证明了所提出方法的优越性。
  
1.  Remote Sensing Object Tracking With Deep Reinforcement Learning Under Occlusion  
  [[Paper]](https://ieeexplore.ieee.org/document/9492311)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing , 2021  
  摘要：目标跟踪是遥感领域中空间地球观测的重要研究方向。尽管现有的基于相关滤波器和深度学习的目标跟踪算法取得了巨大的成功，但对于目标遮挡的问题仍然不尽人意。由于背景复杂变化和跟踪镜头的偏移，遮挡导致目标信息丢失，从而导致检测遗漏。传统上，大多数目标遮挡下的跟踪方法采用复杂的网络模型，重新检测遮挡的目标。为解决这个问题，我们提出了一种全新的目标跟踪方法。首先，构建了一个基于深度强化学习（DRL）的动作决策-遮挡处理网络（AD-OHNet），实现了在目标遮挡下具有低计算复杂度的目标跟踪。其次，采用时空上下文、目标外观模型和运动矢量来提供遮挡信息，驱动强化学习中的动作，在完全遮挡下提高跟踪的准确性并同时保持速度。最后，我们在吉林一号商业遥感卫星拍摄的波哥大、香港和圣地亚哥三个遥感视频数据集上评估了所提出的AD-OHNet。这些视频数据集都存在空间分辨率低、背景杂乱和小目标等问题。实验结果验证了所提出跟踪器是有效且高效的。
  
1.  Single Object Tracking in Satellite Videos: Deep Siamese Network Incorporating an Interframe Difference Centroid Inertia Motion Model  
  [[Paper]](https://www.mdpi.com/2072-4292/13/7/1298)  No code  
  发表刊物: Remote Sens, 2021  
  摘要：卫星视频单目标跟踪引起了广泛关注。地球观测技术中遥感平台的发展使得获取高分辨率的卫星视频变得越来越方便，极大地加快了地面目标的跟踪速度。然而，大图像中目标尺寸较小、多个移动目标之间相似度高以及目标与背景之间的区分度较差使得这一任务变得极具挑战性。为了解决这些问题，本文提出了一种深度孪生网络（DSN）结合帧间差异质心惯性运动（ID-CIM）模型的方法。在目标跟踪任务中，DSN内部包括一个模板分支和一个搜索分支；它从这两个分支提取特征，并利用孪生区域建议网络在搜索分支中获取目标的位置。ID-CIM机制被提出来以减轻模型漂移。这两个模块构建了ID-DSN框架，并相互增强最终的跟踪结果。此外，我们还采用现有的用于遥感图像的目标检测数据集生成适用于卫星视频单目标跟踪的训练数据集。在从国际空间站和“吉林一号”卫星获取的六个高分辨率卫星视频上进行了消融实验。将所提出的ID-DSN结果与其他11个最先进的跟踪器进行了比较，包括不同的网络和主干。比较结果显示，我们的ID-DSN在单个NVIDIA GTX1070Ti GPU上实现了0.927的精确度和0.694的成功度，每秒处理帧数（FPS）为32.117。
  
1.  Rotation adaptive correlation filter for moving object tracking in satellite videos  
  [[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231221000862)  No code  
  发表刊物: Neurocomputing, 2021  
  摘要：作为一种新的地球观测方法，视频卫星可以为目标跟踪提供高时间分辨率的遥感图像。卫星视频中的目标跟踪在计算机视觉领域具有很大的潜力，但也面临挑战。尽管已经提出了许多卫星视频目标跟踪算法，但没有一个解决了跟踪旋转物体的问题。由于卫星视频通常采用正视图，物体的旋转在其中非常普遍，这个问题亟需解决。因此，在本文中，我们提出了一种适应旋转的相关滤波（RACF）跟踪算法来解决物体旋转引起的问题。该算法提供了以下改进：(a)提出了一种估计物体旋转角度的方法，以在物体旋转期间保持特征图的稳定。这种方法可以克服基于梯度直方图（HOG）的跟踪器在卫星视频中无法处理物体旋转的缺点；(b)使算法能够估计物体旋转引起的边界框尺寸的变化。实验结果表明，我们的算法可以在吉林一号卫星的六个视频中以99.84%的精度和92.96%的成功率跟踪目标。
  
1.  Remote sensing target tracking in satellite videos based on a variable-angle-adaptive Siamese network  
  [[Paper]](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12170)  No code  
  发表刊物: IET Image Processing, 2021  
  摘要：卫星视频中的遥感目标跟踪在各个领域中起着关键作用。然而，由于卫星视频序列的复杂背景和高动态目标的多次旋转变化，常规的自然场景目标跟踪方法不能直接用于此类任务，并且很难保证其鲁棒性和准确性。为了解决这些问题，本文提出了一种基于可变角度自适应Siamese网络（VAASN）的卫星视频中的遥感目标跟踪算法。具体而言，该方法基于全卷积Siamese网络（Siamese-FC）。首先，在特征提取阶段，为了减少复杂背景的影响，我们提出了一种新的多频特征表示方法，并将OctConv引入AlexNet架构中，以适应新的特征表示。然后，在跟踪阶段，为了适应目标旋转的变化，引入了一种可变角度自适应模块，利用单个深度神经网络（TextBoxes++）的快速文本检测器从模板帧和检测帧中提取角度信息，并对检测帧进行角度一致性更新操作。最后，使用卫星数据集进行定性和定量实验，结果表明所提出的方法可以提高跟踪准确性，并实现高效率。
  
1.  Object Tracking in Satellite Videos Based on Convolutional Regression Network With Appearance and Motion Features  
  [[Paper]](https://ieeexplore.ieee.org/document/8994098)  No code  
  发表刊物: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2020  
  摘要：目标跟踪是计算机视觉众多应用中最重要的组成部分之一。商业卫星提供的遥感视频使得将目标跟踪扩展到地球观测领域成为可能。在卫星视频中，诸如车辆和飞机之类的典型移动目标只覆盖了少数像素区域，很容易与周围复杂的地面场景混淆。由于分辨率的限制，卫星视频中附近的相似目标很难通过外观细节区分。因此，由于干扰引起的跟踪漂移也是一个棘手的问题。面对这些挑战，传统的基于手工设计视觉特征的相关滤波器方法在卫星视频中取得了不理想的结果。基于深度神经网络的方法在各种普通视觉跟踪基准上展示了其优势，但其在卫星视频上的结果尚未被探索。本文将深度学习技术应用于卫星视频中的目标跟踪，以获得更好的性能。采用简单的回归网络，将回归模型与卷积层和梯度下降算法相结合。回归网络充分利用丰富的背景上下文来学习鲁棒的跟踪器。与手工设计的特征不同，采用了预训练的深度神经网络提取的外观特征和运动特征，以实现精确的目标跟踪。在跟踪器遇到模糊的外观信息时，运动特征可以提供互补和区分性信息，以提高跟踪性能。对各种卫星视频的实验结果表明，所提出的方法在跟踪性能上优于其他现有方法。

1.  Small Target Tracking in Satellite Videos Using Background Compensation  
  [[Paper]](https://ieeexplore.ieee.org/document/9044613)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2020  
  摘要：通过视频技术，卫星可以检测动态目标并分析其运动特征。卫星视频中的目标跟踪可以提取关键地面目标的动态信息，用于目标监测和轨迹预测。目标跟踪算法受到目标运动特性（如速度和方向）以及背景特性（如光照变化、遮挡和与目标相似的背景）的影响。然而，这些问题在卫星视频摄像头中很少被研究。由于卫星视频中目标的纹理和颜色特征较差，当前的跟踪算法不适用于卫星视频。因此，在本文中，我们通过两个方面来增强卫星视频技术中的目标跟踪：1）样本训练策略和2）样本表征。我们建立了一个针对目标和背景的滤波器训练机制，以提高跟踪算法的区分能力。然后，我们使用Gabor滤波器构建目标特征模型，增强目标和背景之间的对比度。此外，我们提出了一个跟踪状态评估指标，以避免跟踪漂移。使用九组吉林一号卫星视频进行的跟踪实验表明，所提出的方法可以在目标属性较弱的情况下准确定位目标。因此，本文为使用卫星视频技术实现更强大的跟踪做出了贡献。
  
1.  Object Tracking on Satellite Videos: A Correlation Filter-Based Tracking Method With Trajectory Correction by Kalman Filter  
  [[Paper]](https://ieeexplore.ieee.org/document/8809377)  No code  
  发表刊物: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019  
  摘要：在处理卫星视频中的目标跟踪问题时，面临着许多挑战，例如目标的小尺寸、纹理缺乏、与背景的相似性等。在本文中，我们提出了一种基于高速相关滤波器（CF）的卫星视频目标跟踪器。该跟踪器利用卫星视频中移动目标的全局运动特征来约束跟踪过程，使用卡尔曼滤波器（KF）来修正移动目标的跟踪轨迹。因此，我们的跟踪器被命名为CFKF。此外，设计了一个跟踪置信度模块，将信息从基于CF的位置检测器传递给基于KF的轨迹校正器，并研究了优化的模型更新频率以加快跟踪器的速度并提高其性能。此外，在跟踪过程中可以利用基于斜率计算的方向检测器获得目标的方向。在卫星视频数据集上进行的实验证明，我们的跟踪器CFKF在准确性和鲁棒性方面优于其他代表性的基于CF的跟踪方法，并且速度快。

1.  Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations  
  [[Paper]](https://ieeexplore.ieee.org/document/8880656)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2019  
  摘要：作为地球观测的新方法，视频卫星能够通过提供高时空分辨率的遥感图像，连续监测地球表面上的特定事件。视频观测使得一系列新的卫星应用成为可能，例如目标跟踪和道路交通监测。在本文中，我们解决了卫星视频中快速目标跟踪的问题，通过开发一种基于相关滤波器和运动估计的新型跟踪算法：基于核相关滤波器（KCF）。所提出的算法提供了以下改进：1）通过结合卡尔曼滤波器和运动轨迹平均的新颖运动估计算法，减轻了KCF的边界效应，2）解决了当移动物体部分或完全遮挡时跟踪失败的问题。实验结果表明，我们的算法可以在卫星视频中以95%的准确率跟踪移动物体。
  
1.  Tracking Objects From Satellite Videos: A Velocity Feature Based Correlation Filter  
  [[Paper]](https://ieeexplore.ieee.org/document/8736008)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2019  
  摘要：卫星视频目标跟踪是遥感领域的一个新课题，指的是实时从卫星视频中跟踪感兴趣的移动物体。感兴趣的目标通常只占据卫星视频图像中的几个像素，即使是较长的列车。因此，与传统的视觉跟踪相比，卫星视频目标跟踪仍面临新的挑战，包括低分辨率目标的检测、特征表示较少以及背景与目标极为相似等。对于卫星视频目标跟踪，研究较少，我们对现有的跟踪算法在卫星视频数据上的适用性知之甚少。本文首次深入研究了13种传统视觉跟踪中的典型跟踪器。实验结果表明，大多数最先进的跟踪算法主要依赖亮度、颜色特征或卷积特征，并且由于特征表示不足，它们无法跟踪卫星视频中的目标。为了克服这一困难，我们提出了一种速度相关滤波（VCF）算法，该算法利用速度特征和惯性机制（IM）构建卫星视频目标跟踪的特定核相关滤波器。速度特征具有高度的判别能力，可以检测卫星视频中的移动目标，而惯性机制可以自适应地防止模型漂移。在三个真实的卫星视频数据集上的实验结果表明，VCF在准确度和成功率方面优于最先进的跟踪方法，并且速度超过每秒100帧。
  
1.  Can We Track Targets From Space? A Hybrid Kernel Correlation Filter Tracker for Satellite Video  
  [[Paper]](https://ieeexplore.ieee.org/document/8789388)  No code  
  发表刊物: IEEE Transactions on Geoscience and Remote Sensing, 2019  
  摘要：尽管基于相关滤波器的跟踪器在视觉跟踪中取得了巨大的成功，但它们是否仍然适用于通过位于地球上方非常高的卫星或空间站获取的卫星视频数据是存疑的。困难在于，与超过一百万像素的图像大小相比，目标通常只占据几个像素，并且几乎融入了相似的背景中。由于相关滤波器模型强烈依赖于特征的质量和跟踪目标的空间布局，它们在卫星视频跟踪任务中可能会失败。在本文中，我们提出了一种混合核相关滤波器（HKCF）跟踪器，在岭回归框架中自适应地使用两种互补的特征。一种特征是光流，可以检测目标的变化像素。另一种特征是梯度方向直方图，可以捕捉目标的轮廓和纹理信息。同时提出了一种自适应融合策略来在不同的卫星视频中利用这两种特征的优势。我们对六个真实卫星视频数据集进行了定量评估。结果表明，我们的方法在每秒超过100帧的速度下优于最先进的跟踪方法。
  
1.  Object Tracking in Satellite Videos Based on a Multiframe Optical Flow Tracker  
  [[Paper]](https://ieeexplore.ieee.org/document/8735957)  No code  
  发表刊物: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019  
  摘要：目标跟踪是计算机视觉中的热门课题。随着超高分辨率（VHR）遥感技术的兴起，现在可以在卫星视频中跟踪感兴趣的目标。然而，由于卫星视频中的目标与整个图像相比通常太小，并且与背景过于相似，大多数最先进的算法无法以满意的准确性跟踪目标。鉴于光流在检测目标的微小运动方面具有巨大潜力，我们提出了一种用于卫星视频目标跟踪的多帧光流跟踪器。将Lucas-Kanade光流方法与HSV颜色系统和积分图像相结合，用于在卫星视频中跟踪目标，而多帧差分方法在光流跟踪器中用于更好地解释目标。对五个VHR遥感卫星视频数据集进行的实验证明，与最先进的目标跟踪算法相比，所提出的方法可以更准确地跟踪目标。
  
1.  Object Tracking in Satellite Videos by Fusing the Kernel Correlation Filter and the Three-Frame-Difference Algorithm  
  [[Paper]](https://ieeexplore.ieee.org/document/8225723)  No code  
  发表刊物: IEEE Geoscience and Remote Sensing Letters, 2017  
  摘要：目标跟踪是计算机视觉领域的一个热门课题。通过超高分辨率遥感传感器提供的详细空间信息，可以在卫星视频中跟踪感兴趣的目标。近年来，相关滤波器在目标跟踪方面取得了有希望的结果。然而，在处理卫星视频中的目标跟踪时，由于每个目标的尺寸与整个图像相比太小，并且目标与背景非常相似，核相关滤波器（KCF）跟踪器的效果较差。因此，在本文中，我们提出了一种新的卫星视频目标跟踪方法，将KCF跟踪器与三帧差分算法融合。本文提出了一种特定的策略，利用KCF跟踪器和三帧差分算法构建一个强大的跟踪器。我们在三个卫星视频中评估了所提出的方法，并展示了其优于其他最先进的跟踪方法的优势。


### MOT
### 运动目标检测
