import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)


from example.sr.msanomalydetector import SpectralResidual
from example.sr.msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode
from example import ADdataset, Preprocessor, Evaluator, ReSummary, get_logger


def detect_anomaly(series, threshold, mag_window, score_window, sensitivity, detect_mode, batch_size=-1):
    """
    SR model.

    Params
    ------
    series: DataFrame, time series data including `timestamp` and `value`.
    threshold: float, the threshold of anomaly score.
    mag_window: int, the size of q.
    score_window: int, the size of z.
    sensitivity: int, anomaly confidence coefficient.
    detect_mode: DetectMode, detect mode.
    batch_size: int, sliding window size.

    Returns
    -------
    DataFrame: model results including `timestamp`, `value`, `mag`, `score`, `isAnomaly`.
    """
    detector = SpectralResidual(series=series, threshold=threshold, mag_window=mag_window, score_window=score_window,
                                sensitivity=sensitivity, detect_mode=detect_mode, batch_size=batch_size)
    result = detector.detect()
    return result

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SR anomaly detection")
    parser.add_argument("--dataset", type=str, default="KPI", choices=["KPI"], help="dataset name")
    parser.add_argument("--datadir", type=str, default="data", help="data dirname")
    parser.add_argument("--logdir", type=str, default="log", help="log dirname")
    parser.add_argument("--method", type=int, default=0, help="evaluation method")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 读取参数
    args = parse_args()
    # 建立日志
    logging_dir = os.path.join(project_dir, args.logdir)
    logger = get_logger(logging_path=logging_dir, logging_name="utsad_sr")
    # 读取数据集
    data_dir = os.path.join(project_dir, args.datadir)
    dataset = ADdataset(root=data_dir, dataset=args.dataset)
    # 记录结果
    res = ReSummary()
    # 异常检测
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        # 预处理
        pre = Preprocessor(test_df, test_label)
        test_df, test_label, test_missing = pre.process()
        # SR模型
        df = detect_anomaly(test_df, THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99, DetectMode.anomaly_only)
        # 模型评测
        evaluator = Evaluator(df["score"].values, test_label, test_missing, method=args.method)
        result, best_th = evaluator.evaluate()
        logger.info("metric: {}, precision: {:.4f}, recall: {:.4f}, best-f1: {:.4f}".format(name, result["precision"], result["recall"], result["best-f1"]))
        res.add(evals={name: result}, results={"y_true": evaluator.label[evaluator.missing==0], "y_pred": evaluator.pred[evaluator.missing==0]})
        # break
    # 评测汇总
    all_res, all_result = res.summary()
    logger.info("dataset: {}, evaluation method: {}, average precision: {:.4f}, average recall: {:.4f}, average best f1: {:.4f}, all precision: {:.4f}, all recall: {:.4f}, all best-f1: {:.4f}".format(args.dataset, args.method, all_res["precision"], all_res["recall"], all_res["best-f1"], all_result["precision"], all_result["recall"], all_result["best-f1"]))