""" Monitoring data reading and preprocessing """
import os
import pandas as pd
import numpy as np
import logging
import time
import json


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
        if dataset not in ["KPI", "Yahoo", "AWS"]:
            raise ValueError("dataset {} is not available".format(dataset))

    def __iter__(self):
        """
        Returns
        -------
        str: kpi type.
        (DataFrame, DataFrame) or (None, None): training data with columns [`timestamp`, `value`] and labels with column `label`.
        (DataFrame, DataFrame): test data with columns [`timestamp`, `value`] and labels with column `label`.
        """
        if self.dataset == "KPI":
            import uuid
            # data_dir = os.path.join(self.root, "KPI")
            data_dir = os.path.join(self.root, "KPI-Anomaly-Detection/Finals_dataset")
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
                if test_label["label"].sum() == 0:
                    continue
                yield str(kpi), (train_df, train_label), (test_df, test_label)
        elif self.dataset == "Yahoo":
            data_dir = os.path.join(self.root, "Yahoo")
            # groups = ["A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"]
            groups = ["A1Benchmark"]
            for group in groups:
                tmp_dir = os.path.join(data_dir, group)
                files = sorted(os.listdir(tmp_dir))
                for file in files:
                    if "json" in file or "all" in file:
                        continue
                    df = pd.read_csv(os.path.join(tmp_dir, file))
                    if df.shape[1] == 3:
                        test_df = df[["timestamp", "value"]]
                        test_label = df[["is_anomaly"]].rename(columns={"is_anomaly": "label"})
                    else:
                        test_df = df[["timestamps", "value"]].rename(columns={"timestamps": "timestamp"})
                        test_label = df[["anomaly"]].rename(columns={"anomaly": "label"})
                    if test_label["label"].sum() == 0:
                        continue
                    if test_df["timestamp"].max() < 1e9:
                        test_df["timestamp"] = test_df["timestamp"]*3600 + 1416726000
                    yield file.split(".")[0], (None, None), (test_df, test_label)
        elif self.dataset == "AWS":
            data_dir = os.path.join(self.root, "AWS")
            label1 = json.load(open(os.path.join(data_dir, "combined_labels.json"), "r"))
            label2 = json.load(open(os.path.join(data_dir, "combined_windows.json"), "r"))
            filelist = [file for file in os.listdir(data_dir) if file.endswith("csv")]
            for file in filelist:
                df = pd.read_csv(os.path.join(data_dir, file))
                df["label"] = 0
                for ts1 in label1[file]:
                    if df[df["timestamp"]==ts1].shape[0] > 0:
                        df.loc[df["timestamp"]==ts1, "label"] = 1
                df["timestamp"] = df["timestamp"].apply(str_to_ts)
                for t1, t2 in label2[file]:
                    df.loc[(df["timestamp"]>=str_to_ts(t1)) & (df["timestamp"]<=str_to_ts(t2)), "label"] = 1
                test_df = df[["timestamp", "value"]]
                test_label = df[["label"]]
                if test_label["label"].sum() == 0:
                    continue
                yield file.split(".")[0], (None, None), (test_df, test_label)
        

class Preprocessor:
    """
    A data preprocessor which preprocesses the raw data including data normalization and missing data imputation.

    Params
    ------
    data: DataFrame, monitoring time series data.
    label: DataFrame, data labels.
    normalize: bool, if do data normalization(z-score) or not. 
    fillna: None or str, missing data imputation method.
    If set None, zeros will be inserted at missing points.
    If set `prediction`, adopt the estimation method proposed by the paper `Time-Series Anomaly Detection Service at Microsoft`(https://arxiv.org/pdf/1906.03821).
    If set `interpolation`, linear interpolation is utilized for missing points.
    smooth: float, the top proportion of extreme data to be removed for better time series clustering.
    """
    def __init__(self, data, label, normalize=False, fillna=None, smooth=0.0):
        if data.shape[0] != label.shape[0]:
            raise ValueError("the size of data is not consistent with the size of label, with data size {} and label size {}".format(data.shape[0], label.shape[0]))
        self.data = data
        self.label = label
        self.normalize = normalize
        if fillna and fillna not in ["prediction", "interpolation"]:
            raise ValueError("the fillna method should be `prediction` or `interpolation`, the input {} is not available".format(fillna))
        self.fillna = fillna
        self.smooth = smooth

    def process(self):
        """
        data preprocessing.

        Returns
        -------
        data: DataFrame, monitoring time series data after preprocessing.
        label: 1-D ndarray, data labels which add missing data label with 0.
        missing: 1-D ndarray, missing data labels with 0(not missing) and 1(missing). 
        """
        data, label, missing = self.complete_ts()
        if self.fillna:
            data["value"] = self.data_imputation(data["value"].values, missing)
        if self.normalize:
            values, mean, std = normalize_kpi(data["value"].values, excludes=missing)
            data["value"] = values
        if self.smooth:
            data["value"] = self.smoothing_extreme_values(data["value"].values)

        return data, label, missing

    def complete_ts(self):
        """
        complete the data according to the given `timestamp`.

        Returns
        -------
        DataFrame: complete time series data.
        1-D ndarray: data labels which add missing data label with 0.
        1-D ndarray: missing data labels with 0(not missing) and 1(missing). 
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
        Missing data imputation.
        For `prediction` method, estimate the missing values by sum up the slope of the last value with previous values.
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j).
        For `interpolation` method, linear interpolation is utilized for missing points.

        Params
        ------
        values: 1-D ndarray, time series values.
        missing: 1-D ndarray, missing data labels.
        window: int, the number of preceding values for `prediction` method.

        Returns
        -------
        1-D ndarray: new values after data imputation.
        """
        new_values = values.copy()
        if self.fillna == "prediction":
            missing_index = np.arange(len(missing))[missing == 1]
            for index in missing_index:
                if index < window + 1:
                    continue
                grads = new_values[index-1] - new_values[index - window - 1:index - 1]
                dt = range(window, 0, -1)
                mean_grads = np.mean(grads/dt)
                new_values[index] = new_values[index-window] + mean_grads*window
        elif self.fillna == "interpolation":
            missing_index = np.arange(len(missing))[missing == 1]
            pos_index = np.arange(len(missing))[missing == 0]
            pos_values = values[pos_index]
            neg_values = np.interp(missing_index, pos_index, pos_values)
            new_values[missing_index] = neg_values

        return new_values

    def smoothing_extreme_values(self, values):
        """
        In general, the ratio of anomaly points in a time series is less than 5%.
        As such, simply remove the top 5% data which deviate the most from the mean 
        value, and use linear interpolation to fill them.
        
        Params
        ------
        values: 1-D ndarray, a time series.
        
        Returns
        -------
        new_values: 1-D ndarray, the smoothed `values`.
        """
        if self.normalize:
            new_values = values.copy()
        else:
            new_values, mean, std = normalize_kpi(values)
        
        # get the deviation of each point from zero mean
        values_deviation = np.abs(new_values)
        
        # the abnormal portion
        abnormal_portion = self.smooth
        
        # replace the abnormal points with linear interpolation
        abnormal_max = np.max(values_deviation)
        abnormal_index = np.argwhere(values_deviation >= abnormal_max * (1-abnormal_portion))
        abnormal = abnormal_index.reshape(len(abnormal_index))
        normal_index = np.argwhere(values_deviation < abnormal_max * (1-abnormal_portion))
        normal = normal_index.reshape(len(normal_index))
        normal_values = new_values[normal]
        abnormal_values = np.interp(abnormal, normal, normal_values)
        new_values[abnormal] = abnormal_values
        
        return new_values

    def extract_baseline(self, values, w=120):
        """
        A simple but effective method for removing noises if to apply moving 
        average with a small sliding window(`w`) on the KPI(`values`),separating 
        its curve into two parts:baseline and residuals.
        For a KPI,T,with a sliding window of length of `w`,stride = 1,for each 
        point x(t),the corresponding point on the baseline,denoted as x(t)*,is the 
        mean of vector (x(t-w+1),...,x(t)).
        Then the diffrence between x(t) and x(t)* is called a residuals.
        
        Params
        ------
        values: 1-D ndarray, time series after preprocessing and smoothing.
        w: int, the length of sliding window. 
            
        Returns
        -------
        new_values: 1-D ndarray, the baseline of rawdata.
        residuals: 1-D ndarray, the residuals between rawdata and baseline.
        """
        # moving average to get the baseline
        baseline = np.convolve(values, np.ones((w,))/w, mode="valid")
        # get the residuals, the difference between raw series and baseline
        residuals = values[w-1:] - baseline
        new_values, mean, std = normalize_kpi(baseline)
        
        return new_values, residuals

def normalize_kpi(values, mean=None, std=None, excludes=None):
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

def statistics(dataset):
    """
    print the statistics info of given dataset.

    Params
    ------
    dataset: ADdataset, given time series dataset.
    """
    all_num = [0, 0]
    anomalies = [0, 0]
    missing_num = [0, 0]
    num = [0, 0]
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        logger.info("kpi {} data".format(name))
        if train_df is not None:
            train_pre = Preprocessor(train_df, train_label)
            all_train_df, all_train_label, train_missing = train_pre.process()
            logger.info("train dataset, data: {}, missing rate: {:.4f}, anomaly rate: {:.4f}".format(all_train_df.shape[0], float(train_missing.sum())/all_train_df.shape[0], float(train_label.label.sum())/all_train_df.shape[0]))
            all_num[0] += all_train_df.shape[0]
            anomalies[0] += train_label.label.sum()
            missing_num[0] += train_missing.sum()
            num[0] += 1

        test_pre = Preprocessor(test_df, test_label)
        all_test_df, all_test_label, test_missing = test_pre.process()
        logger.info("test dataset, data: {}, missing rate: {:.4f}, anomaly rate: {:.4f}".format(all_test_df.shape[0], float(test_missing.sum())/all_test_df.shape[0], float(test_label.label.sum())/all_test_df.shape[0]))
        all_num[1] += all_test_df.shape[0]        
        anomalies[1] += test_label.label.sum()
        missing_num[1] += test_missing.sum()
        num[1] += 1

    logger.info("all {} data".format(dataset.dataset))
    logger.info("train dataset, type: {}, data: {}, missing rate: {:.4f}, anomaly rate: {:.4f}".format(num[0], all_num[0], float(missing_num[0])/(all_num[0]+1e-6), float(anomalies[0])/(all_num[0]+1e-6)))
    logger.info("test dataset, type: {}, data: {}, missing rate: {:.4f}, anomaly rate: {:.4f}".format(num[1], all_num[1], float(missing_num[1])/(all_num[1]+1e-6), float(anomalies[1])/(all_num[1]+1e-6)))

def str_to_ts(x):
    """
    convert timestr to timestamp.

    Params
    ------
    x: str, time string with format `%Y-%m-%d %H:%M:%S`.

    Returns
    -------
    int: timestamp.
    """
    return int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))

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

if __name__ == "__main__":
    dataset_name = "AWS"
    # 建立日志
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    logging_dir = os.path.join(project_dir, "log")
    logger = get_logger(logging_path=logging_dir, logging_name="utsad_dataset_{}".format(dataset_name))
    # 读取数据集
    data_dir = os.path.join(project_dir, "data")
    dataset = ADdataset(root=data_dir, dataset=dataset_name)
    statistics(dataset)
