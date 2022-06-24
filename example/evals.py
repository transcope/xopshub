""" Model evaluation """
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def get_score(y_true, y_pred):
    """
    calculate the evaluation metrics.

    Params
    ------
    y_true: 1-D ndarray, true labels.
    y_pred: 1-D ndarray, model predictions.

    Returns
    -------
    dict: evaluation metrics including precision, recall and f1.
    """
    result = {}
    result["precision"] = precision_score(y_true, y_pred)
    result["recall"] = recall_score(y_true, y_pred)
    result["best-f1"] = f1_score(y_true, y_pred)
    return result

class Evaluator:
    """
    A model evaluator which includes several evaluation methods.

    Params
    ------
    score: 1-D ndarray, model prediction scores.
    label: 1-D ndarray, true labels.
    missing: 1-D ndarray or None, missing labels.
    pred: 1-D ndarray or None, model predictions.
    method: int(>=-1), evaluation method. 
    If -1, model predictions (or scores) are directly used for evaluation.
    If 0, we adopt a point-adjust strategy following `Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications`(https://arxiv.org/pdf/1802.03903.pdf).
    If >0, we adopt the evaluation method of the AIOps competition(https://github.com/iopsai/iops/blob/master/evaluation/evaluation.py), and the method value means the delay (the competition sets 7).
    is_higher_better: bool, if higher scores mean higher probabilities of anomaly or not.
    """
    def __init__(self, score, label, missing=None, pred=None, method=-1, is_higher_better=True):
        if len(score) != len(label):
            raise ValueError("score and label do not have the same size, with score size {} and label size {}.".format(len(score), len(label)))
        if is_higher_better:
            self.score = score
        else:
            self.score = -1.0 * score
        self.label = label
        if missing is None:
            self.missing = np.zeros_like(self.label)
        else:
            if len(missing) != len(label):
                raise ValueError("missing and label do not have the same size, with missing size {} and label size {}.".format(len(missing), len(label)))
            self.missing = missing

        if pred is not None and len(pred) != len(label):
            raise ValueError("pred and label do not have the same size, with pred size {} and label size {}.".format(len(pred), len(label)))
        self.pred = pred
        if not isinstance(method, int) or method < -1:
            raise ValueError("parameter method should be -1 or a positive interger.")
        self.method = method

    def evaluate(self):
        """
        Evaluation results.

        Returns
        -------
        dict: evaluation metrics including precision, recall and best-f1.
        float or None: best threshold.
        """
        if self.pred is None:
            result, threshold, self.pred = self.bf_search()
            return result, threshold
        else:
            if self.method != -1:
                self.pred, _ = self.adjust_predicts()

            result = get_score(self.label[self.missing==0], self.pred[self.missing==0])
            return result, None

    def bf_search(self, step_num=1000, display_freq=50, verbose=False):
        """
        Find the best-f1 score by searching best threshold within given steps.

        Params
        ------
        step_num: int, search steps.
        display_freq: int, display frequency.
        verbose: bool, if print the intermediate results or not.

        Returns
        -------
        dict: evaluation metrics including precision, recall and best-f1.
        float: the threshold for best-f1.
        1-D ndarray: the predictions for best-f1.
        """
        desc_score_indices = np.argsort(self.score[self.missing == 0], kind="mergesort")[::-1]
        y_score = self.score[self.missing == 0][desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_score.size - 1]

        if len(threshold_idxs) < step_num:
            step = 1
        else:
            step = len(threshold_idxs)//step_num
        search_range = list(range(0, len(threshold_idxs), step))
        if len(threshold_idxs)-1 not in search_range:
            search_range += [len(threshold_idxs)-1]
        if verbose:
            print("search range: {}-{}, search step: {}, search times: {}, max search: {}".format(y_score[threshold_idxs[0]], y_score[threshold_idxs[-1]], step, len(search_range), len(threshold_idxs)))

        best = {"best-f1": -1.}
        best_th = 0.0
        preds = None
        for i in search_range:
            threshold = y_score[threshold_idxs[i]]
            target, predict = self.calc_score(threshold)
            if target["best-f1"] > best["best-f1"]:
                best_th = threshold
                best = target
                preds = predict
            if verbose and i % display_freq == 0:
                print("cur thr: ", threshold, str(target), str(best), best_th)
        return best, best_th, preds

    def calc_score(self, threshold):
        """
        Calculate f1 score according to the given threshold.
        
        Params
        ------
        threshold: float, the threshold of anomaly score. A point is labeled as "anomaly" if its score is higher than the threshold.
        
        Returns
        -------
        dict: evaluation metrics.
        1-D ndarray: the predictions.
        """
        if self.method == -1:
            predict = (self.score >= threshold).astype("int")
            t = get_score(self.label[self.missing == 0], predict[self.missing == 0])
        else:
            predict, latency = self.adjust_predicts(threshold)
            t = get_score(self.label[self.missing == 0], predict[self.missing == 0])
            if latency is not None:
                t["latency"] = latency
        
        return t, predict
        
    def adjust_predicts(self, threshold=None):
        """
        Calculate adjusted predict labels.

        Params
        ------
        threshold: float or None, the threshold of anomaly score.
        A point is labeled as "anomaly" if its score is higher than the threshold.
        If None, `pred` will be used.

        Returns
        -------
        1-D ndarray: predict labels.
        float or None: the average latency of the point-adjust method.
        """
        score = np.asarray(self.score)
        label = np.asarray(self.label)
        
        if self.pred is None:
            if threshold is None:
                raise ValueError("Both pred and threshold are not given!")
            predict = score >= threshold
        else:
            predict = self.pred > 0.1
        
        if self.method == 0:
            latency = 0
            actual = label > 0.1
            anomaly_state = False
            anomaly_count = 0
            for i in range(len(score)):
                if actual[i] and predict[i] and not anomaly_state:
                        anomaly_state = True
                        anomaly_count += 1
                        for j in range(i, 0, -1):
                            if not actual[j]:
                                break
                            else:
                                if not predict[j]:
                                    predict[j] = True
                                    latency += 1
                elif not actual[i]:
                    anomaly_state = False
                if anomaly_state:
                    predict[i] = True
            
            return predict.astype("int"), latency / (anomaly_count + 1e-4)
        else:
            predict = predict.astype("int")
            splits = np.where(label[1:] != label[:-1])[0] + 1
            is_anomaly = label[0] == 1
            new_predict = np.array(predict)
            pos = 0

            for sp in splits:
                if is_anomaly:
                    if 1 in predict[pos:min(pos + self.method + 1, sp)]:
                        new_predict[pos: sp] = 1
                    else:
                        new_predict[pos: sp] = 0
                is_anomaly = not is_anomaly
                pos = sp
            sp = len(label)

            if is_anomaly:
                if 1 in predict[pos: min(pos + self.method + 1, sp)]:
                    new_predict[pos: sp] = 1
                else:
                    new_predict[pos: sp] = 0

            return new_predict, None
        
class ReSummary:
    """
    The result summary for different models.
    """
    def __init__(self, metrics=["precision", "recall", "best-f1"]):
        self.metrics = metrics
        self.eval_lists = []
        self.result = {"y_pred": [], "y_true": []}

    def add(self, evals, results):
        for name, eval in evals.items():
            for metric in self.metrics:
                if metric not in eval:
                    raise ValueError("{} is missing in the input evaluation results".format(metric))
            df = pd.DataFrame(eval, index=[name])
            self.eval_lists.append(df)

        if "y_pred" not in results or "y_true" not in results:
            raise ValueError("missing input results")

        for k in self.result.keys():
            self.result[k] += [results[k]]

    def summary(self):
        res = pd.concat(self.eval_lists, axis=0)
        all_res = {metric: res[metric].mean() for metric in self.metrics}
        if "best-f1" in all_res:
            all_res["best-f1"] = 2*all_res["precision"]*all_res["recall"]/(all_res["precision"]+all_res["recall"])
        y_true_all = np.concatenate(self.result["y_true"], axis=0)
        y_pred_all = np.concatenate(self.result["y_pred"], axis=0)
        all_result = get_score(y_true_all, y_pred_all)
        return all_res, all_result

    def save(self, file):
        res = pd.concat(self.eval_lists, axis=0)
        res.to_csv(file)