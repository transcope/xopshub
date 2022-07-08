""" Opprentice model """
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)


import numpy as np
import pandas as pd
from example.opprentice.time_series_detector.feature_service import extract_features
from example.opprentice.time_series_detector.utils import DEFAULT_WINDOW
from example import ADdataset, Preprocessor, Evaluator, ReSummary, get_logger
from sklearn.ensemble import RandomForestClassifier
import pickle


def _check_features(featlist):
    """
    check feature results.

    Params
    ------
    featlist: list, lists of feature lists.

    Returns
    -------
    bool: if the feature sizes are all consistent and not equal 0, return True. 
    """
    length = [len(feat) for feat in featlist]
    return len(set(length)) == 1 and set(length) != {0}

class FeatureExtractor:
    """
    A feature extractor which calculates the features according to the specific sliding windows of the time series.
    It includes three time series, i.e., dataA, dataB and dataC, where dataA is the target data, dataB is the short-term history data, and dataC is the long-term history data.

    Params
    ------
    window: int, the size of sliding windows.
    intervalC: int, the long-term time interval(days) before the target timestamp.
    intervalB: int, the short-term time interval(days) before the target timestamp.
    tstype: str, the time interval of the given data, 1 minute, 5 minute or 1 hour.
    """
    def __init__(self, window=DEFAULT_WINDOW, intervalC=7, intervalB=1, tstype="1min"):
        self.window = window
        if tstype == "1min":
            self.deltatimeC = intervalC*1440
            self.deltatimeB = intervalB*1440
        elif tstype == "5min":
            self.deltatimeC = intervalC*288
            self.deltatimeB = intervalB*288
        elif tstype == "hour":
            self.deltatimeC = intervalC*24
            self.deltatimeB = intervalB*24
        else:
            raise ValueError("tstype must be '1min', '5min' or 'hour', but got {}".format(tstype))

    def _get_features(self, df):
        """
        calculate all features of the given time series. If the short-term history data is not available, the features will not be calculated.

        Params
        ------
        df: DataFrame, time series data including columns `timestamp` and `value`.

        Returns
        -------
        feats: 2-D ndarray, shape of (nsamples, nfeatures), where nsamples is less than the input data size.
        """
        featlist = []
        for i in range(df.shape[0]):
            ts = i - self.window
            tsB = i - self.deltatimeB
            tsB_min = tsB - self.window
            tsB_max = tsB + self.window
            if tsB_min < 0:
                continue
            tsC = i - self.deltatimeC
            tsC_min = tsC - self.window
            tsC_max = tsC + self.window
            dataA = df.loc[ts:i, "value"].values
            dataB = df.loc[tsB_min:tsB_max, "value"].values
            if tsC_min < 0:
                dataC = dataB
            else:
                dataC = df.loc[tsC_min:tsC_max, "value"].values
            series_data = list(np.concatenate([dataC, dataB, dataA]))
            features = extract_features(series_data, self.window)
            featlist.append(features)
            
        if not _check_features(featlist):
            raise ValueError("the size of features are not consistent")
        feats = np.asarray(featlist, dtype="float")
        return feats

    def get_features(self, train_df, train_label, test_df, test_label):
        """
        generate the train and test dataset with various features.

        Params
        ------
        train_df: DataFrame, train data with column `timestamp` and `value`.
        train_label: 1-D ndarray, train label.
        test_df: DataFrame, test data with column `timestamp` and `value`.
        test_label: 1-D ndarray, test label.

        Returns
        -------
        train_X: 2-D ndarray, train features with shape (nsamples, nfeatures).
        train_Y: 1-D ndarray, train label.
        test_X: 2-D ndarray, test features with shape (nsamples, nfeatures).
        test_Y: 1-D ndarray, test label.
        """
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        feature = self._get_features(df)
        logger.info("feature size: {}".format(feature.shape))
        train_X = feature[:-test_df.shape[0]]
        train_Y = train_label[df.shape[0]-feature.shape[0]:]
        test_X = feature[-test_df.shape[0]:]
        test_Y = test_label
        return train_X, train_Y, test_X, test_Y

def parse_args():
    """
    Arg parser.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Opprentice anomaly detection")
    parser.add_argument("--dataset", type=str, default="KPI", choices=["KPI"], help="dataset name")
    parser.add_argument("--datadir", type=str, default="data", help="data dirname")
    parser.add_argument("--logdir", type=str, default="log", help="log dirname")
    parser.add_argument("--method", type=int, default=0, help="evaluation method")
    parser.add_argument("--save", action="store_true", default=False, help="save model results")

    # exception handling
    try:
        args = parser.parse_args()
    except:
        print("help message, or you can use `main.py -h` for more details")
        print("--dataset dataset_name, it should be in [KPI] and default value is KPI")
        print("--datadir data_dirname, default value is data")
        print("--logdir log_dirname, default value is log")
        print("--method evaluation_method, it should be an interger no less than -1 and default value is 0")
        print("--save, it is used to save the model results")
        parser.error("wrong usage")
    return args

if __name__ == "__main__":
    # 读取参数
    args = parse_args()
    # 建立日志
    logging_dir = os.path.join(project_dir, args.logdir)
    logger = get_logger(logging_path=logging_dir, logging_name="utsad_opprentice")
    # 读取数据集
    data_dir = os.path.join(project_dir, args.datadir)
    dataset = ADdataset(root=data_dir, dataset=args.dataset)
    # 记录结果
    res = ReSummary()
    prediction = {}
    # 间隔时间参数
    ts_type = {60: "1min", 300: "5min", 1: "hour"}
    # 异常检测
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        logger.info("read metric {} data".format(name))
        # 预处理
        train_pre = Preprocessor(train_df, train_label, fillna=None)
        train_df, train_label, train_missing = train_pre.process()
        test_pre = Preprocessor(test_df, test_label, fillna=None)
        test_df, test_label, test_missing = test_pre.process()
        logger.info("preprocessing completed")
        # 计算特征
        time_interval = int(train_df["timestamp"].diff().min())
        fe = FeatureExtractor(tstype=ts_type[time_interval])
        train_X, train_Y, test_X, test_Y = fe.get_features(train_df, train_label, test_df, test_label)
        logger.info("extracting feature successfully, train size: {}, train label: {}, test_size: {}, test label: {}".format(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape))
        # np.savez_compressed(os.path.join(project_dir, "result", "{}.npz".format(name)), train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)
        # 机器学习模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        num = train_df.shape[0] - train_X.shape[0]
        non_missing = train_missing[num:]==0
        model.fit(train_X[non_missing], train_Y[non_missing])
        score = model.predict_proba(test_X)[:, 1]
        logger.info("model training, score size: {}".format(score.shape))
        prediction[name] = score
        # 模型评测
        evaluator = Evaluator(score, test_label, test_missing, method=args.method)
        result, best_th = evaluator.evaluate()
        logger.info("metric: {}, precision: {:.4f}, recall: {:.4f}, best-f1: {:.4f}".format(name, result["precision"], result["recall"], result["best-f1"]))
        res.add(evals={name: result}, results={"y_true": evaluator.label[evaluator.missing==0], "y_pred": evaluator.pred[evaluator.missing==0]})
        # break
    # 评测汇总
    all_res, all_result = res.summary()
    logger.info("dataset: {}, evaluation method: {}, average precision: {:.4f}, average recall: {:.4f}, average best f1: {:.4f}, all precision: {:.4f}, all recall: {:.4f}, all best-f1: {:.4f}".format(args.dataset, args.method, all_res["precision"], all_res["recall"], all_res["best-f1"], all_result["precision"], all_result["recall"], all_result["best-f1"]))
    if args.save:
        save_dir = os.path.join(project_dir, "visualization/result")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "opprentice_prediction_{}.pkl".format(args.dataset)), "wb") as f:
            pickle.dump(prediction, f)