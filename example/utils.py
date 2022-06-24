""" Monitoring data reading and preprocessing """
import os
import pandas as pd
import numpy as np
import logging
import time


class ADdataset:
    """
    A data generator which generates both training and testing data from the given datasets.
    
    Params
    ------
    root: str, the data file path.
    dataset: str, the dataset name.
    """
    def __init__(self, root, dataset):
        self.root = root
        self.dataset = dataset
        if dataset not in ["SMD", "KPI", "ASD"]:
            raise ValueError("dataset {} is not available".format(dataset))

    def __iter__(self):
        """
        Returns
        -------
        str: kpi type.
        (DataFrame, DataFrame): training data with columns [`timestamp`, `value`] and labels with column `label`.
        (DataFrame, DataFrame): test data with columns [`timestamp`, `value`] and labels with column `label`.
        """
        if self.dataset == "KPI":
            import uuid
            data_dir = os.path.join(self.root, "KPI")
            all_train_df = pd.read_csv(os.path.join(data_dir, "phase2_train.csv"))
            all_train_df["KPI ID"] = all_train_df["KPI ID"].apply(lambda x: uuid.UUID(x))
            all_test_df = pd.read_hdf(os.path.join(data_dir, "phase2_ground_truth.hdf"))
            kpilist = all_train_df["KPI ID"].unique().tolist()
            for kpi in kpilist:
                train_data = all_train_df[all_train_df["KPI ID"] == kpi].sort_values(by=["timestamp"], ascending=True).reset_index(drop=True)
                train_df = train_data[["timestamp", "value"]]
                train_label = train_data[["label"]]
                test_data = all_test_df[all_test_df["KPI ID"] == kpi].sort_values(by=["timestamp"], ascending=True).reset_index(drop=True)
                test_df = test_data[["timestamp", "value"]]
                test_label = test_data[["label"]]
                yield str(kpi), (train_df, train_label), (test_df, test_label)
        

class Preprocessor:
    """
    A data preprocessor which preprocesses the raw data including data normalization and missing data imputation.

    Params
    ------
    data: DataFrame, monitoring time series data.
    label: DataFrame, data labels.
    normalize: bool, if do data normalization(z-score) or not. 
    fillna: bool, if do missing data imputation or not, which estimates the missing value according to the method proposed by the paper `Time-Series Anomaly Detection Service at Microsoft`(https://arxiv.org/pdf/1906.03821).
    """
    def __init__(self, data, label, normalize=False, fillna=True):
        if data.shape[0] != label.shape[0]:
            raise ValueError("the size of data is not consistent with the size of label, with data size {} and label size {}".format(data.shape[0], label.shape[0]))
        self.data = data
        self.label = label
        self.normalize = normalize
        self.fillna = fillna

    def process(self):
        """
        data preprocessing.

        Returns
        -------
        data: DataFrame, monitoring time series data after preprocessing.
        label: 1-D ndarray, data labels which add missing data label with 0.
        missing: 1-D ndarray, missing data labels with 0(missing) and 1(not missing). 
        """
        data, label, missing = self.complete_ts()
        if self.normalize:
            values, mean, std = self.normalize_kpi(data["value"].values, excludes=missing)
            data["value"] = values
        if self.fillna:
            data["value"] = self.data_imputation(data["value"].values, missing)

        return data, label, missing

    def complete_ts(self):
        """
        complete the data according to the given `timestamp`.

        Returns
        -------
        DataFrame: complete time series data.
        1-D ndarray: data labels which add missing data label with 0.
        1-D ndarray: missing data labels with 0(missing) and 1(not missing). 
        """
        df = pd.concat([self.data, self.label], axis=1)
        df["missing"] = 0
        timestamp = np.asarray(df["timestamp"].values, np.int64)
        intervals = np.unique(np.diff(timestamp))
        interval = np.min(intervals)
        if interval == 0:
            raise ValueError("Duplicated values in `timestamp`")
        for itv in intervals:
            if itv % interval != 0:
                raise ValueError("Not all intervals in `timestamp` are multiples of the minimum interval")

        tmp = pd.DataFrame(np.arange(timestamp[0], timestamp[-1] + interval, interval, dtype=np.int64), columns=["timestamp"])
        df = pd.merge(tmp, df, on=["timestamp"], how="left")
        df["value"] = df["value"].fillna(0)
        df["missing"] = df["missing"].fillna(1)
        df["label"] = df["label"].fillna(0)

        return df[["timestamp", "value"]], df["label"].values, df["missing"].values

    def data_imputation(self, values, missing, window=5):
        """
        Estimate the missing values by sum up the slope of the last value with previous values.
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j).

        Params
        ------
        values: 1-D ndarray, time series values.
        missing: 1-D ndarray, missing data labels.
        window: int, the number of preceding values.

        Returns
        -------
        1-D ndarray: new values after data imputation.
        """
        new_values = values.copy()
        missing_index = np.arange(len(missing))[missing == 1]
        for index in missing_index:
            if index < window + 1:
                continue
            grads = new_values[index-1] - new_values[index - window - 1:index - 1]
            dt = range(window, 0, -1)
            mean_grads = np.mean(grads/dt)
            new_values[index] = new_values[index-window] + mean_grads*window

        return new_values

    def normalize_kpi(self, values, mean=None, std=None, excludes=None):
        """
        z-score normalization.
        Params
        ------
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not :obj:`None`, will use this `mean` to standardize
            `values`. If :obj:`None`, `mean` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        std (float): If not :obj:`None`, will use this `std` to standardize
            `values`. If :obj:`None`, `std` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)

        Returns
        -------
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
        """
        values = np.asarray(values, dtype=np.float32)
        if len(values.shape) != 1:
            raise ValueError('`values` must be a 1-D array')
        if (mean is None) != (std is None):
            raise ValueError('`mean` and `std` must be both None or not None')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            if excludes.shape != values.shape:
                raise ValueError('The shape of `excludes` does not agree with '
                                'the shape of `values` ({} vs {})'.
                                format(excludes.shape, values.shape))

        if mean is None:
            if excludes is not None:
                val = values[np.logical_not(excludes)]
            else:
                val = values
            mean = val.mean()
            std = val.std()

        return (values - mean) / std, mean, std

def get_logger(logging_path, logging_name, logging_level=logging.INFO):
    """
    add a logger.
    """
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    logging_file = os.path.join(logging_path, logging_name + "_" + time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + ".log")
    logger = logging.getLogger(logging_name)
    logging.basicConfig(filename=logging_file, level=logging_level, format="[%(asctime)s][%(levelname)s][%(funcName)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    return logger