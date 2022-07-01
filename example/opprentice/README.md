# Opprentice异常检测算法
这是Opprentice异常检测算法的使用说明文档。

## 算法简介
Opprentice是一种适用于单维时间序列的有监督异常检测方法，也是一套利用机器学习进行异常监控的系统。它的核心异常检测算法采用了14种异常侦测算子生成133个不同的特征，然后训练随机森林模型进行异常检测，具体可见论文[Opprentice: Towards Practical and Automatic Anomaly Detection Through Machine Learning](http://netman.cs.tsinghua.edu.cn/wp-content/uploads/2015/11/liu_imc15_Opprentice.pdf)。本算例是Opprentice异常检测算法的近似版本，采用了其中部分特征，共包含三类特征（统计特征、拟合特征、分类特征），特征维度90+，算例实现参考了腾讯开源的[metis系统](https://github.com/tencent/metis)。

## 环境配置

### 依赖包
* numpy>=1.15.2

* pandas>=0.25.3

* scikit-learn>=0.20.0

* tsfresh>=0.11.1

### 安装

* 执行以下命令安装所有依赖：

```
pip install -r requirements.txt
```

## 运行算例
* 在已有数据集上运行SR算法：

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

    ```
    python main.py --method 7
    ```

## 算例结果
* 评测方案1

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.6597|0.4936|0.5647|0.6015|0.5661|0.5833|

* 评测方案2

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.8605|0.8758|0.8681|0.8925|0.8991|0.8958|

* 评测方案3

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.8026|0.7584|0.7799|0.7153|0.7679|0.7406|