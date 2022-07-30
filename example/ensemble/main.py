#-*- encoding: utf-8 -*-
""" ensemble model """
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)


import time
import datetime
import pandas as pd
import numpy as np
import example.ensemble.models as models
from example import ADdataset, Preprocessor, Evaluator, ReSummary, get_logger
import pickle


def ensemble_trainer(train_df, test_df, freq="D", voting_num=3):
    """
    anomaly detection through ensembling.

    Params
    ------
    train_df: DataFrame, train data with column `timestamp` and `value`.
    test_df: DataFrame, test data with column `timestamp` and `value`.
    freq: str, detection period, "H" (hour), "D" (day), "M" (month) or "Y" (year).
    voting_num: int, the number of methods for voting to detect anomalies. 

    Returns
    -------
    1-D ndarray: prediction.
    """
    if train_df is None:
        df = test_df.copy()
        df["test"] = 1
    else:
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        df["test"] = np.concatenate([np.zeros(train_df.shape[0]), np.ones(test_df.shape[0])])
    df["timestamp"] = df["timestamp"].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    # 去重
    if df["timestamp"].unique().shape[0] != df.shape[0]:
        df = df.groupby(["timestamp"]).agg({"value": "mean", "test": "max"})
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
    else:
        df = df.set_index(["timestamp"])
        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()

    test_indexes = df[df["test"] == 1].index
    test_preds = []
    for ts in test_indexes:
        history_data = df[(df.index >= ts + datetime.timedelta(hours=7*int(-24))) & (df.index < ts)]["value"]
        check_value = df[df.index == ts]["value"].values[0]
        if len(history_data) < 24:
            preds = "no alarm"
        else:
            preds = voting(history_data, check_value, freq, voting_num)
        if preds == "no alarm":
            test_preds.append(0)
        else:
            test_preds.append(1)
    
    if len(test_indexes) != len(test_preds):
        raise ValueError("length of preds is not consistent with length of data")

    return np.array(test_preds)


def voting(data, check_value, freq, voting_num):
    """
    voting ensembling methods.

    Params
    ------
    data: Series, history data of time series.
    check_value: float, value of detecting step.
    freq: str, detection period, "H" (hour), "D" (day), "M" (month) or "Y" (year).
    voting_num: int, the number of methods for voting to detect anomalies. 

    Returns
    -------
    str: detection results, "uprush", "anticlimax" or "no alarm".
    """
    check_result_list = {}
    # 支持的模型
    model_list = ["pop", "amplitude", "tail", "iforest", "fitting"]
    for i in range(len(model_list)):
        # print(model_list[i])
        alg = models.create(model_list[i], freq)
        result = alg.check(data, check_value)
        check_result_list[model_list[i]] = result

    result_type_list = []
    for i in check_result_list:
        # print("model: %s" %i)
        # print("check result:%s, percent:%f" %(check_result_list[i][0], check_result_list[i][1]))
        if check_result_list[i]:
            result_type_list.append(check_result_list[i][0])
        else:
            result_type_list.append(check_result_list[i])

    if result_type_list.count("uprush") >= voting_num:
        return "uprush"
    elif result_type_list.count("anticlimax") >= voting_num:
        return "anticlimax"
    else:
        return "no alarm"

def parse_args():
    """
    Arg parser.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble anomaly detection")
    parser.add_argument("--dataset", type=str, default="KPI", choices=["KPI", "Yahoo", "AWS"], help="dataset name")
    parser.add_argument("--datadir", type=str, default="data", help="data dirname")
    parser.add_argument("--logdir", type=str, default="log", help="log dirname")
    parser.add_argument("--method", type=int, default=0, help="evaluation method")
    parser.add_argument("--save", action="store_true", default=False, help="save model results")

    # exception handling
    try:
        args = parser.parse_args()
    except:
        print("help message, or you can use `main.py -h` for more details")
        print("--dataset dataset_name, it should be in [KPI, Yahoo, AWS] and default value is KPI")
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
    logger = get_logger(logging_path=logging_dir, logging_name="utsad_ensemble")
    # 读取数据集
    data_dir = os.path.join(project_dir, args.datadir)
    dataset = ADdataset(root=data_dir, dataset=args.dataset)
    # 记录结果
    res = ReSummary()
    prediction = {}
    # with open(os.path.join(project_dir, "visualization/result", "ensemble_prediction_{}.pkl".format(args.dataset)), "rb") as f:
    #     prediction = pickle.load(f)
    # 异常检测
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        logger.info("read metric {} data".format(name))
        # 预处理
        if train_df is not None:
            train_pre = Preprocessor(train_df, train_label, fillna=None)
            train_df, train_label, train_missing = train_pre.process()
        test_pre = Preprocessor(test_df, test_label, fillna=None)
        test_df, test_label, test_missing = test_pre.process()
        logger.info("preprocessing completed")
        # 多算法投票
        preds = ensemble_trainer(train_df, test_df)
        logger.info("model prediction, pred size: {}".format(preds.shape))
        prediction[name] = preds
        # 模型评测
        # preds = prediction[name]
        evaluator = Evaluator(preds, test_label, test_missing, preds, method=args.method)
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
        with open(os.path.join(save_dir, "ensemble_prediction_{}.pkl".format(args.dataset)), "wb") as f:
            pickle.dump(prediction, f)