# 核密度估计算法的探索和实践
如何使用KDE算法（主要是高斯核密度估计）对监控数据进行分析。

## 算法
核密度估计方法，用于估计未知的密度函数，不依赖数据的先验知识和假设，属于非参数估计方法。
 * 公式如下所示：
 
 $$ \begin{aligned}
     \hat{f}_{h}(x)=\frac{1}{n} \sum_{i=1}^{n} K_{h}\left(x-x_{i}\right)=\frac{1}{n h} \sum_{i=1}^{n} K\left(\frac{x-x_{i}}{h}\right) 
 \end{aligned} $$

 * 主要参数包括：
   * 核密度函数： $ K(.) $ 为核函数， 这里主要采用高斯核函数；
   * 带宽：$ h > 0 $ ，反映了KDE曲线整体的平坦程度，（用人类语言解释，必要时加图）
      * 当带宽越大，KDE整体曲线就越平坦，说明观察到的数据点在最终形成的曲线形状中所占比重越小。

## 数据
这里使用的是AWS监控指标数据集中的CPU使用率，具体参见 <a href="../data/README.md">数据说明.</a>

## 实验
这里包括三组实验，分别是：1. 有效性分析；2. 参数影响评估；3. 泛化性测试。

* 有效性分析
  * 这里采用数据集（ec2_cpu_utilization_24ae8d.csv），该数据集包括两个异常点，分布情况如下图所示；
  * 采用高斯核函数，带宽选择0.15；
  * 如下图所示，可以看到两个异常点明显突出，说明算法能够侦测到指标异常。
 

  

<p align="center">
  <img src="../image/kde-application/Ori.png" width="600"/>
  </br>图1.1：原始数据集
</p>
<p align="center">
  <img src="../image/kde-application/bw_0.15.png" width="600"/>
  </br>图1.2：bw = 0.15
</p>



* 参数影响评估
  * 延用上个实验的数据集；
  * 这里KDE的带宽分别采用0.15、0.5、1、2、5、10；
  * 如下图所示，带宽越小，异常点越明显，也就是在当前数据集中，观察点的数据在曲线形状中占比越大，越容易发现异常。

<p align="center">
  <img src="../image/kde-application/bw_0.15.png" width="600"/>
  </br>图2.1：bw = 0.15
</p>
<p align="center">
  <img src="../image/kde-application/bw_0.5.png" width="600"/>
  </br>图2.2：bw = 0.5
</p>
<p align="center">
  <img src="../image/kde-application/bw_1.png" width="600"/>
  </br>图2.3：bw = 1
</p>
<p align="center">
  <img src="../image/kde-application/bw_2.png" width="600"/>
  </br>图2.4：bw = 2
</p>
<p align="center">
  <img src="../image/kde-application/bw_5.png" width="600"/>
  </br>图2.5：bw = 5
</p>
<p align="center">
  <img src="../image/kde-application/bw_10.png" width="600"/>
  </br>图2.6：bw = 10
</p>



<!-- 可以看出，在这个数据集中，，也就是相对来说不同的数据集适合的带宽也不相同。
 -->

* 泛化性测试
  * 使用AWS剩下四个数据集进行测试；
  * 不同数据集的KDE带宽参数不同；
  * 如下图所示，可以看到：KDE能够侦测到不同节点上的指标异常。
<p align="center">
  <img src="../image/kde-application/bw1.png" width="600"/>
  </br>图3.1：数据集：ec2_cpu_utilization_ac20cd.csv，bw = 0.15
</p>
<p align="center">
  <img src="../image/kde-application/bw2.png" width="600"/>
  </br>图3.2：数据集：ec2_cpu_utilization_5f5533.csv，bw = 0.5
</p>
<p align="center">
  <img src="../image/kde-application/bw3.png" width="600"/>
  </br>图3.3：数据集：ec2_cpu_utilization_825cc2.csv，bw = 1
</p>
<p align="center">
  <img src="../image/kde-application/bw4.png" width="600"/>
  </br>图3.4：数据集：ec2_cpu_utilization_fe7f93.csv，bw = 2
</p>

