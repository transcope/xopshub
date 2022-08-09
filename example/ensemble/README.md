# 集成类异常检测算法
这是集成类异常检测算法的使用说明文档。

## 算法简介
集成类异常检测算法是一种适用于单维时间序列的无监督异常检测方法，通过多种算法投票进行异常检测。它主要包括基于统计特征（同比、环比、振幅）的检测方法、基于拟合特征（指数权重移动平均）的检测方法和孤立森林算法，具体可见文章[时间序列异常检测机制的研究](https://zhuanlan.zhihu.com/p/35544112)。本算例实现参考了360公司开源的[aiopstools项目](https://github.com/jixinpu/aiopstools/tree/master/aiopstools/anomaly_detection)。

## 环境配置

### 依赖包
* numpy

* pandas

* scikit-learn

### 安装

* 执行以下命令安装所有依赖：

```
pip install -r requirements.txt
```

## 运行算例
* 在已有数据集上运行集成算法：

```
python main.py --dataset dataset_name --datadir data_dir --logdir log_dir --method eval_method --save
```

使用方法：main.py [-h] [--dataset DATASET] [--datadir DATADIR] [--logdir LOGDIR] [--method METHOD] [--save]

`-h` 打印帮助信息并退出

`--dataset` 数据集名称，默认为KPI

`--datadir` 数据集所在目录名称，默认为data

`--logdir` 结果日志文件保存目录名称，默认为log

`--method` 评测方法参数，默认为0

`--save` 保存模型预测结果

例如，全部采用默认值运行：

```
python main.py
```

* [不同评测方案](https://github.com/transcope/xopshub/tree/main/example/README.md)的运行方法：

    * 评测方案1

    ```
    python main.py --method -1
    ```

    * 评测方案2

    ```
    python main.py --method 0
    ```

    * 评测方案3

    KPI、AWS：
    ```
    python main.py --dataset KPI --method 7
    ```
    Yahoo：
    ```
    python main.py --dataset Yahoo --method 3
    ```

## 算例结果
* 评测方案1

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.0515|0.3477|0.0896|0.0553|0.2987|0.0934|
|Yahoo|0.0310|0.7422|0.0595|0.0291|0.6321|0.0556|
|AWS|0.1295|0.3299|0.1860|0.1194|0.3187|0.1737|

* 评测方案2

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.1329|0.9493|0.2331|0.1587|0.9617|0.2724|
|Yahoo|0.0463|0.9936|0.0885|0.0451|0.9964|0.0863|
|AWS|0.3144|0.9792|0.4760|0.2939|0.9786|0.4520|

* 评测方案3

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.1271|0.9072|0.2229|0.1488|0.8917|0.2551|
|Yahoo|0.0463|0.9936|0.0885|0.0451|0.9964|0.0863|
|AWS|0.2463|0.8125|0.3780|0.2497|0.7827|0.3786|