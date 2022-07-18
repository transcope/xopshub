# PUAD异常检测算法
这是PUAD异常检测算法的使用说明文档。

## 算法简介
PUAD是一种适用于大量单维时间序列的半监督异常检测方法，基于正样本无标签学习（PU learning）和少量标签信息进行异常检测。它的主要步骤包括KPI聚类（clustering），聚类中心KPI进行正样本无标签学习（PU learning），以及其余KPI根据聚类中心学习标签进行半监督学习（semi-supervised learning），具体可见论文[Robust KPI Anomaly Detection for Large-Scale Software Services with Partial Labels](https://netman.aiops.org/wp-content/uploads/2021/12/paper-ISSRE21-PUAD.pdf)。本算例是PUAD异常检测算法的近似版本，其中特征计算采用了Opprentice算例的实现，算例实现参考了[PUAD公开代码](https://github.com/PUAD-code/PUAD)。

## 环境配置

### 依赖包
* numpy

* pandas

* scipy

* scikit-learn

* nlopt

* imbalanced-learn

### 安装

* 执行以下命令安装所有依赖：

```
pip install -r requirements.txt
```

## 运行算例
* 在已有数据集上运行SR算法：

```
python main.py --dataset dataset_name --datadir data_dir --logdir log_dir --method eval_method --cluster --feature --save
```

使用方法：main.py [-h] [--dataset DATASET] [--datadir DATADIR] [--logdir LOGDIR] [--method METHOD] [--cluster] [--feature] [--save]

`-h` 打印帮助信息并退出

`--dataset` 数据集名称，默认为KPI

`--datadir` 数据集所在目录名称，默认为data

`--logdir` 结果日志文件保存目录名称，默认为log

`--method` 评测方法参数，默认为0

`--cluster` 运行Rocka聚类算法，否则会读取当前目录下的聚类结果文件

`--feature` 计算时序特征，否则会读取保存的特征结果文件

`--save` 保存模型预测结果

例如，全部采用默认值运行：

```
python main.py --cluster --feature
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

    ```
    python main.py --method 7
    ```

## 算例结果
* 评测方案1

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.3313|0.3503|0.3405|0.0588|0.4288|0.1035|

* 评测方案2

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.6160|0.6481|0.6317|0.4819|0.7186|0.5769|

* 评测方案3

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.4293|0.5354|0.4765|0.3175|0.5140|0.3925|