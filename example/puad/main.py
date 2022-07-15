#-*- coding:utf-8 -*-
""" PUAD model """
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from example.puad.framework.CPLELearning import CPLELearningModel
from example.puad.framework.PULearning import PULearningModel
from example import ADdataset, Preprocessor, Evaluator, ReSummary, get_logger
from example.opprentice.main import FeatureExtractor
from example.puad.cluster import run_rocka 
import pickle


def cal_ts_feats(datadir, featdir, dataset="KPI"):
    """
    calculate the features of time series data.

    Params
    ------
    datadir: str, the data path.
    featdir: str, the feature path.
    dataset: str, the dataset name.
    """
    data = ADdataset(root=datadir, dataset=dataset)
    # 间隔时间参数
    ts_type = {60: "1min", 300: "5min", 1: "hour"}
    for name, (train_df, train_label), (test_df, test_label) in data:
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
        np.savez_compressed(os.path.join(featdir, "{}.npz".format(name)), train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y)

def label_positive(Y, centroid_KPI_label_count):
    """
    select the positive samples randomly.

    Params
    ------
    Y: 1-D ndarray, the real labels.
    centroid_KPI_label_count: int, the number of initial positive samples to select.

    Returns
    -------
    label: 1-D ndarray, the initial label for PU-learning, which consists of certain positive samples (1) and massive unlabeled samples (-1).
    """
    label = -np.ones_like(Y, dtype=int)
    positive_index = np.arange(len(Y))[Y == 1]
    if len(positive_index) <= centroid_KPI_label_count:
        label[positive_index] = 1
    else:
        sample_index = np.random.choice(positive_index, size=centroid_KPI_label_count, replace=False)
        label[sample_index] = 1

    return label

def centroid_pu_learning(X, Y, centroid_KPI_label_count, speed=1000):
    """
    PU-learning for centroid kpi.

    Params
    ------
    X: 2-D ndarray, the feature vector.
    Y: 1-D ndarray, the real labels.
    centroid_KPI_label_count: int, the number of initial positive samples to select.
    speed: int, the number of newly added unlabeled samples for each iteration.

    Returns
    -------
    PU_labels: 1-D ndarray, the labels after PU-learning.
    """
    real_label = Y.copy()
    Y_source = label_positive(Y, centroid_KPI_label_count)
    # Start PU learning
    PU_model = PULearningModel(X, Y_source, len(Y_source))
    PU_model.pre_training(0.2)
    PU_labels, positive_label_count = PU_model.add_reliable_samples_using_RandomForest(0.04, speed, 0.7, real_label)
    return PU_labels

def cple_learning(centroid_X, centroid_Y, X=None, Y=None):
    """
    semi-supervised learning method CPLE.

    Params
    ------
    centroid_X: 2-D ndarray, the feature vector of centroid kpi.
    centroid_Y: 1-D ndarray, the labels of centroid kpi after pu-learning.
    X: None or 2-D ndarray, the feature vector of non-centroid kpi.
    Y: None or 1-D ndarray, the unlabeled labels of non-centroid kpi.

    Returns
    -------
    model: RandomForestClassifier, the classifier.
    """
    if X is not None and Y is not None:
        train_X = np.concatenate([centroid_X, X])
        train_Y = np.concatenate([centroid_Y, Y])
    else:
        train_X = centroid_X
        train_Y = centroid_Y

    model = CPLELearningModel(basemodel=RandomForestClassifier(n_estimators=100), max_iter=50, predict_from_probabilities=True, real_label=None)
    model.fit(train_X, train_Y)

    return model

def pu_trainer(datadir, clustering_df, dataset="KPI"):
    """
    PU-learning for centroid kpi.

    Params
    ------
    datadir: str, the data path.
    clustering_df: dataframe, the clustering result.
    dataset: str, the dataset name.

    Returns
    -------
    pu_results: dict, the results of pu-learning.
    """
    data = ADdataset(root=datadir, dataset=dataset)
    centroidlist = clustering_df[clustering_df["centroid"] != -1]["kpi"].tolist()
    pu_results = {}
    for name, (train_df, train_label), (test_df, test_label) in data:
        if name in centroidlist:
            train_pre = Preprocessor(train_df, train_label, fillna=None)
            train_df, train_label, train_missing = train_pre.process()
            cluster_id = clustering_df[clustering_df["kpi"]==name]["cluster"].values[0]
            logger.info("PU Learning, metric {}, cluster {}".format(name, cluster_id))
            features = np.load(os.path.join(feat_dir, "{}.npz".format(name)))
            train_X = features["train_X"]
            train_Y = features["train_Y"]
            num = train_df.shape[0] - train_X.shape[0]
            non_missing = train_missing[num:]==0
            centroid_X = train_X[non_missing]
            centroid_Y = train_Y[non_missing]
            if len(features["train_Y"]) > 50000:
                PU_labels = centroid_pu_learning(centroid_X, centroid_Y, 20, 10000)
            else:
                PU_labels = centroid_pu_learning(centroid_X, centroid_Y, 20, 1000)
            logger.info("Finish PU learning for centroid: {}".format(str(Counter(PU_labels))))
            pu_results[name] = (centroid_X, PU_labels)
            # break
    return pu_results

def parse_args():
    """
    Arg parser.
    """
    import argparse
    parser = argparse.ArgumentParser(description="PUAD anomaly detection")
    parser.add_argument("--dataset", type=str, default="KPI", choices=["KPI"], help="dataset name")
    parser.add_argument("--datadir", type=str, default="data", help="data dirname")
    parser.add_argument("--logdir", type=str, default="log", help="log dirname")
    parser.add_argument("--method", type=int, default=0, help="evaluation method")
    parser.add_argument("--cluster", action="store_true", default=False, help="run clustering model")
    parser.add_argument("--feature", action="store_true", default=False, help="calculate kpi features")
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
        print("--cluster, it is used to run clustering model")
        print("--feature, it is used to calculate kpi features")
        print("--save, it is used to save the model results")
        parser.error("wrong usage")
    return args

if __name__ == "__main__":
    # 读取参数
    args = parse_args()
    # 建立日志
    logging_dir = os.path.join(project_dir, args.logdir)
    logger = get_logger(logging_path=logging_dir, logging_name="utsad_puad")
    # 读取数据集
    data_dir = os.path.join(project_dir, args.datadir)
    # 记录结果
    res = ReSummary()
    prediction = {}
    # KPI聚类
    if args.cluster:
        clustering_df = run_rocka(datadir=args.datadir, dataset=args.dataset, minPts=4)
        clustering_df.to_csv(os.path.join(project_dir, "example/puad", "rocka.csv"), index=False)
    else:
        clustering_df = pd.read_csv(os.path.join(project_dir, "example/puad", "rocka.csv"))
    # 计算特征
    feat_dir= os.path.join(project_dir, "result")
    if args.feature:
        cal_ts_feats(datadir=args.datadir, featdir=feat_dir, dataset=args.dataset)
    # PU Learning
    pu_results = pu_trainer(data_dir, clustering_df, dataset=args.dataset)
    logger.info("Finish PU-Learning")
    if args.save:
        with open(os.path.join(project_dir, "example/puad", "pu_results_{}.pkl".format(args.dataset)), "wb") as f:
            pickle.dump(pu_results, f)
    
    # with open(os.path.join(project_dir, "example/puad", "pu_results_{}.pkl".format(args.dataset)), "rb") as f:
        # pu_results = pickle.load(f)

    # 半监督学习
    dataset = ADdataset(root=data_dir, dataset=args.dataset)
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        cluster_id = clustering_df[clustering_df["kpi"]==name]["cluster"].values[0]
        centroid_name = clustering_df[clustering_df["centroid"]==cluster_id]["kpi"].values[0]
        logger.info("read metric {} data, cluster: {}, centroid: {}".format(name, cluster_id, centroid_name))
        # 读取特征
        features = np.load(os.path.join(feat_dir, "{}.npz".format(name)))
        # 预处理
        test_pre = Preprocessor(test_df, test_label, fillna=None)
        test_df, test_label, test_missing = test_pre.process()
        test_X = features["test_X"]
        test_Y = features["test_Y"]
        centroid_X, PU_labels = pu_results[centroid_name]
        if name == centroid_name:
            model = cple_learning(centroid_X, PU_labels)
        else:
            train_pre = Preprocessor(train_df, train_label, fillna=None)
            train_df, train_label, train_missing = train_pre.process()
            train_X = features["train_X"]
            train_Y = features["train_Y"]
            new_train_Y = -np.ones_like(train_Y)
            num = train_df.shape[0] - train_X.shape[0]
            non_missing = train_missing[num:]==0
            model = cple_learning(centroid_X, PU_labels, train_X[non_missing], new_train_Y[non_missing])
        logger.info("Finish CPLE-Learning")
        score = model.predict_proba(test_X)[:, 1]
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
        with open(os.path.join(save_dir, "puad_prediction_{}.pkl".format(args.dataset)), "wb") as f:
            pickle.dump(prediction, f)
    
