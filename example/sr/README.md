# SR异常检测算法
这是SR异常检测算法的使用说明文档。

## 算法简介
SR（Spectral Residual）算法是一种适用于单维时间序列的无监督异常检测算法，也是2018AIOps比赛冠军方案的基础算法。它通过傅里叶变换和逆变换得到时间序列的显著图（saliency map），并以该显著图的值与其移动平均的相对差值作为异常分数，具体可见论文 [Time-Series Anomaly Detection Service at Microsoft](https://arxiv.org/pdf/1906.03821)。本算例主要参考了微软开源的 [anomalydetector 项目](https://github.com/microsoft/anomalydetector)。

## 环境配置

### 依赖包
* python == 3.7

* Cython>=0.29.2

* numpy>=1.18.1

* pandas>=0.25.3

### 安装

* 执行以下命令安装所有依赖：

```
pip install -r requirements.txt
```

* 如果python版本不是3.7，需要重新编译Cython程序 `msanomalydetector/._anomaly_kernel_cython.pyx` 生成动态链接库（.so文件），具体可按以下步骤执行：

    * 切换目录

    ```
    cd msanomalydetector
    ```

    * 新建脚本 `setup.py`，具体代码如下：

    ```
    # setup.py
    from distutils.core import setup, Extension
    from Cython.Build import cythonize
    import numpy as np

    setup(ext_modules = cythonize(Extension(
        '_anomaly_kernel_cython',
        sources=['_anomaly_kernel_cython.pyx'],
        language='c',
        include_dirs=[np.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[],
        extra_link_args=[]
    )))
    ```

    * 执行命令：

    ```
    python setup.py build_ext --inplace
    ```

## 运行算例
* 在已有数据集上运行SR算法：

```
python main.py --dataset dataset_name --datadir data_dir --logdir log_dir --method eval_method --save --magwindow mag_window --scorewindow score_window --slidingwindow sliding_window
```

使用方法：main.py [-h] [--dataset DATASET] [--datadir DATADIR] [--logdir LOGDIR] [--method METHOD] [--save] [--magwindow MAGWINDOW] [--scorewindow SCOREWINDOW] [--slidingwindow SLIDINGWINDOW]

`-h` 打印帮助信息并退出

`--dataset` 数据集名称，默认为KPI

`--datadir` 数据集所在目录名称，默认为data

`--logdir` 结果日志文件保存目录名称，默认为log

`--method` 评测方法参数，默认为0

`--save` 保存模型预测结果

`--magwindow` 模型参数，计算SR的均值滤波窗口，默认为3

`--scorewindow` 模型参数，计算显著图异常分数的均值滤波窗口，默认为40

`--slidingwindow` 模型参数，模型输入傅里叶变换的滑动时间窗口，默认为1440

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

* 不同数据集的参数设置：

    * KPI

    ```
    python main.py
    ```

    * Yahoo

    ```
    python main.py --dataset Yahoo --slidingwindow 64
    ```

    * AWS

    ```
    python main.py --dataset AWS --slidingwindow 288
    ```

## 算例结果
* 评测方案1

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.3106|0.1615|0.2125|0.0839|0.1922|0.1169|
|Yahoo|0.5105|0.4420|0.4738|0.1284|0.5411|0.2075|
|AWS|0.1207|0.7571|0.2082|0.1142|0.7384|0.1977|

* 评测方案2

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.7630|0.7861|0.7744|0.7924|0.8536|0.8218|
|Yahoo|0.7833|0.8642|0.8217|0.7219|0.9213|0.8095|
|AWS|0.9518|1.0000|0.9753|0.9454|1.0000|0.9719|

* 评测方案3

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.7529|0.6680|0.7079|0.7813|0.6768|0.7253|
|Yahoo|0.7358|0.8590|0.7927|0.3848|0.9141|0.5416|
|AWS|0.5240|0.8750|0.6555|0.5289|0.8563|0.6539|