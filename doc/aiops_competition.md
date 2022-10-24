# 运维比赛方案
这里主要参考浦发团队在Aiops比赛中所使用的方案。

## 数据
1. 数据量、数据描述

本次实验的数据集为cloudbed1下的node节点的node-1。

## 方案
目的：故障时，指标是否存在异常。
对象：该分析主要针对节点级的故障。具体地，对故障期间的节点指标进行异常检测。
方案：这里包括三个步骤。
1. 数据采集。
2. 预估。故障发生前一段时间的指标样本点，用于训练KDE模型，估计故障期间的指标样本点。
3. 预警。采用K-Sigma方法，对KDE估计（训练和测试）进行判定，输出异常标识。
<p align="center">
  <img src="../image/aiops_competition_pic/flow.png" width="200"/>
  </br>图：方案流程图
</p>

## 数据采集
由于只有故障发生时间点，没有时间范围，因此数据采集需要设置间隔时间（故障发生前后），确保样本污染。

<p align="center">
  <img src="../image/aiops_competition_pic/case.png" width="600"/>
  </br>图：数据采集示例
</p>

如上图所示，采集参数主要包括，
- 故障发生时间宽度（故障开始，故障结束）
- 抛弃样本时间宽度（故障前，故障后）
- 训练样本采集时间宽度

案例学习：

根据已知故障时间点附近，可以看到

<p align="center">
  <img src="../image/aiops_competition_pic/ab_before.png" width="600"/>
  </br>图：故障时间点前，指标异常示例
</p>

<p align="center">
  <img src="../image/aiops_competition_pic/ab_after.png" width="600"/>
  </br>图：故障时间点后，指标异常示例
</p>

给定故障发生时间段之后，可以看到

<p align="center">
  <img src="../image/aiops_competition_pic/abts_before.png" width="600"/>
  </br>图：故障发生前，指标异常示例
</p>

<p align="center">
  <img src="../image/aiops_competition_pic/abts_after.png" width="600"/>
  </br>图：故障发生后，指标异常示例
</p>

## 时间窗分析
相同时间窗内，不同指标采集的样本个数不同，会对KDE识别产生影响。


## 有效性分析

方案的检测效果不稳定。

<p align="center">
  <img src="../image/aiops_competition_pic/ab1.png" width="600"/>
  </br>图：该指标能够看出明显异常
</p>

<p align="center">
  <img src="../image/aiops_competition_pic/ab3.png" width="600"/>
  </br>图：该指标的异常点不明显
</p>

<p align="center">
  <img src="../image/aiops_competition_pic/ab2.png" width="600"/>
  </br>图：该指标的异常点无法理解
</p> 

## 算法说明

* KDE算法  
此方案，使用KDE算法进行异常检验，具体参见 <a href="../data/Gaussian-KDE-Application.md">核密度估计算法的探索和实践</a>。

* K-sigma算法  
K-sigma算法也是异常检测的一种。

# 参考
1. 浦发方案：<a href="./external/浦智运维战队.pdf">[浦智运维战队]</a>
2. 农行方案：<a href="./external/ABC_AIOPS.pdf">[ABC_AIOPS]</a>
