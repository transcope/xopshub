""" SR model """
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)


from example.sr.msanomalydetector import SpectralResidual
from example.sr.msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode
from example import ADdataset, Preprocessor, Evaluator, ReSummary, get_logger
import pickle


def sr_detect(series, threshold=THRESHOLD, mag_window=MAG_WINDOW, score_window=SCORE_WINDOW, sensitivity=99, 
            detect_mode=DetectMode.anomaly_only, batch_size=-1):
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
    # run args
    parser.add_argument("--dataset", type=str, default="KPI", choices=["KPI", "Yahoo", "AWS"], help="dataset name")
    parser.add_argument("--datadir", type=str, default="data", help="data dirname")
    parser.add_argument("--logdir", type=str, default="log", help="log dirname")
    parser.add_argument("--method", type=int, default=0, help="evaluation method")
    parser.add_argument("--save", action="store_true", default=False, help="save model results")
    # model args
    parser.add_argument("--magwindow", type=int, default=3, help="spectrum average filter size")
    parser.add_argument("--scorewindow", type=int, default=40, help="score average filter size")
    parser.add_argument("--slidingwindow", type=int, default=1440, help="sliding window size")

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
        print("--magwindow mag_window, it should be an interger no less than 3 and default value is 3")
        print("--scorewindow score_window, it should be an interger no less than 3 and default value is 40")
        print("--slidingwindow sliding_window, it should be an interger no less than 12 and default value is 1440")
        parser.error("wrong usage")
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
    prediction = {}
    # with open(os.path.join(project_dir, "visualization/result", "sr_prediction_{}.pkl".format(args.dataset)), "rb") as f:
    #     prediction = pickle.load(f)
    # 异常检测
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        # 预处理
        pre = Preprocessor(test_df, test_label, fillna="prediction")
        test_df, test_label, test_missing = pre.process()
        # SR模型
        df = sr_detect(series=test_df, mag_window=args.magwindow, score_window=args.scorewindow, batch_size=args.slidingwindow)
        prediction[name] = df["score"].values
        # 模型评测
        num = test_df.shape[0] - df.shape[0]  # 初始数据因时间窗口无侦测结果
        evaluator = Evaluator(prediction[name], test_label[num:], test_missing[num:], method=args.method)
        result, best_th = evaluator.evaluate()
        logger.info("metric: {}, precision: {:.4f}, recall: {:.4f}, best-f1: {:.4f}".format(name, result["precision"], result["recall"], result["best-f1"]))
        res.add(evals={name: result}, results={"y_true": evaluator.label[evaluator.missing==0], "y_pred": evaluator.pred[evaluator.missing==0]})
        # break
    # 评测汇总
    all_res, all_result = res.summary()
    logger.info("dataset: {}, evaluation method: {}, average precision: {:.4f}, average recall: {:.4f}, average best f1: {:.4f}, all precision: {:.4f}, all recall: {:.4f}, all best-f1: {:.4f}".format(args.dataset, args.method, all_res["precision"], all_res["recall"], all_res["best-f1"], all_result["precision"], all_result["recall"], all_result["best-f1"]))
    logger.info("model params, mag window: {}, score window: {}, sliding window: {}".format(args.magwindow, args.scorewindow, args.slidingwindow))
    if args.save:
        save_dir = os.path.join(project_dir, "visualization/result")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "sr_prediction_{}.pkl".format(args.dataset)), "wb") as f:
            pickle.dump(prediction, f)