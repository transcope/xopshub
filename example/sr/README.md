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

* 模型参数可在 `msanomalydetector/util.py` 中设置。

## 算例结果
* 评测方案1

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.2124|0.2463|0.2281|0.0527|0.3887|0.0928|

* 评测方案2

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.7006|0.8354|0.7621|0.7264|0.8414|0.7797|

* 评测方案3

|dataset|$\bar{P}$|$\bar{R}$|$\bar{F1}$|$P^{* }$|$R^{* }$|$F1^{* }$|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|KPI|0.6409|0.7034|0.6707|0.6081|0.6354|0.6214|