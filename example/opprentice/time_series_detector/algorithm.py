# -*- coding=utf-8 -*-
"""
Tencent is pleased to support the open source community by making Metis available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import xgboost as xgb
from .utils import DEFAULT_WINDOW, TSD_LACK_SAMPLE, TSD_TRAIN_ERR, TSD_OP_SUCCESS, TSD_CAL_FEATURE_ERR, TSD_READ_FEATURE_FAILED, is_standard_time_series
from .feature_service import extract_features


class Statistic(object):
    """
    In statistics, the 68-95-99.7 rule is a shorthand used to remember the percentage of values
    that lie within a band around the mean in a normal distribution with a width of two, four and
    six standard deviations, respectively; more accurately, 68.27%, 95.45% and 99.73% of the values
    lie within one, two and three standard deviations of the mean, respectively.

    WIKIPEDIA: https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    """

    def __init__(self, index=3):
        """
        :param index: multiple of standard deviation
        :param type: int or float
        """
        self.index = index

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :return: 1 denotes normal, 0 denotes abnormal
        """
        if abs(X[-1] - np.mean(X[:-1])) > self.index * np.std(X[:-1]):
            return 0
        return 1

class Ewma(object):
    """
    In statistical quality control, the EWMA chart (or exponentially weighted moving average chart)
    is a type of control chart used to monitor either variables or attributes-type data using the monitored business
    or industrial process's entire history of output. While other control charts treat rational subgroups of samples
    individually, the EWMA chart tracks the exponentially-weighted moving average of all prior sample means.

    WIKIPEDIA: https://en.wikipedia.org/wiki/EWMA_chart
    """

    def __init__(self, alpha=0.3, coefficient=3):
        """
        :param alpha: Discount rate of ewma, usually in (0.2, 0.3).
        :param coefficient: Coefficient is the width of the control limits, usually in (2.7, 3.0).
        """
        self.alpha = alpha
        self.coefficient = coefficient

    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :return: 1 denotes normal, 0 denotes abnormal
        """
        s = [X[0]]
        for i in range(1, len(X)):
            temp = self.alpha * X[i] + (1 - self.alpha) * s[-1]
            s.append(temp)
        s_avg = np.mean(s)
        sigma = np.sqrt(np.var(X))
        ucl = s_avg + self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        lcl = s_avg - self.coefficient * sigma * np.sqrt(self.alpha / (2 - self.alpha))
        if s[-1] > ucl or s[-1] < lcl:
            return 0
        return 1

class PolynomialInterpolation(object):
    """
    In statistics, polynomial regression is a form of regression analysis in which the relationship
    between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x.

    WIKIPEDIA: https://en.wikipedia.org/wiki/Polynomial_regression
    """

    def __init__(self, threshold=0.15, degree=4):
        """
       :param threshold: The critical point of normal.
       :param degree: Depth of iteration.
        """
        self.degree = degree
        self.threshold = threshold

    def predict(self, X, window=DEFAULT_WINDOW):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :param window: the length of window
        :param type window: int
        :return: 1 denotes normal, 0 denotes abnormal
        """
        x_train = list(range(0, 2 * window + 1)) + list(range(0, 2 * window + 1)) + list(range(0, window + 1))
        x_train = np.array(x_train)
        x_train = x_train[:, np.newaxis]
        avg_value = np.mean(X[-(window + 1):])
        if avg_value > 1:
            y_train = X / avg_value
        else:
            y_train = X
        model = make_pipeline(PolynomialFeatures(self.degree), Ridge())
        model.fit(x_train, y_train)
        if abs(y_train[-1] - model.predict(np.array(x_train[-1]).reshape(1, -1))) > self.threshold:
            return 0
        return 1

class EwmaAndPolynomialInterpolation(object):

    def __init__(self, alpha=0.3, coefficient=3, threshold=0.15, degree=4):
        """
        :param alpha: Discount rate of ewma, usually in (0.2, 0.3).
        :param coefficient: Coefficient is the width of the control limits, usually in (2.7, 3.0).
        :param threshold: The critical point of normal.
        :param degree: Depth of iteration.
        """
        self.alpha = alpha
        self.coefficient = coefficient
        self.degree = degree
        self.threshold = threshold

    def predict(self, X, window=DEFAULT_WINDOW):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :param: window: the length of window
        :param type window: int
        :return: 1 denotes normal, 0 denotes abnormal
        """
        ewma_obj = Ewma(self.alpha, self.coefficient)
        ewma_ret = ewma_obj.predict(X)
        if ewma_ret == 1:
            result = 1
        else:
            polynomial_obj = PolynomialInterpolation(self.threshold, self.degree)
            polynomial_ret = polynomial_obj.predict(X, window)
            result = polynomial_ret
        return result

class IForest(object):
    """
    The IsolationForest 'isolates' observations by randomly selecting a feature and then
    randomly selecting a split value between the maximum and minimum values of the selected feature.

    https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
    """

    def __init__(self,
                 n_estimators=3,
                 max_samples="auto",
                 contamination=0.15,
                 max_feature=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        """
        :param n_estimators: The number of base estimators in the ensemble.
        :param max_samples: The number of samples to draw from X to train each base estimator.
        :param coefficient: The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
        :param max_features: The number of features to draw from X to train each base estimator.
        :param bootstrap: If True, individual trees are fit on random subsets of the training data sampled with replacement. If False, sampling without replacement is performed.
        :param random_state: If int, random_state is the seed used by the random number generator;
                              If RandomState instance, random_state is the random number generator;
                              If None, the random number generator is the RandomState instance used  by `np.random`.
        :param verbose: Controls the verbosity of the tree building process.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_feature = max_feature
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def predict(self, X, window=DEFAULT_WINDOW):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :param window: the length of window
        :param type window: int
        :return: 1 denotes normal, 0 denotes abnormal.
        """
        x_train = list(range(0, 2 * window + 1)) + list(range(0, 2 * window + 1)) + list(range(0, window + 1))
        sample_features = zip(x_train, X)
        clf = IsolationForest(self.n_estimators, self.max_samples, self.contamination, self.max_feature, self.bootstrap, self.n_jobs, self.random_state, self.verbose)
        clf.fit(sample_features)
        predict_res = clf.predict(sample_features)
        if predict_res[-1] == -1:
            return 0
        return 1


class Gbdt(object):
    """
    Gradient boosting is a machine learning technique for regression and classification problems,
    which produces a prediction model in the form of an ensemble of weak prediction models,
    typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do,
    and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

    WIKIPEDIA: https://en.wikipedia.org/wiki/Gradient_boosting
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/')
    DEFAULT_MODEL = MODEL_PATH + "gbdt_default_model"

    def __init__(self, threshold=0.15, n_estimators=300, max_depth=10, learning_rate=0.05):
        """
        :param threshold: The critical point of normal.
        :param n_estimators: The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        :param max_depth: Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.
        :param learning_rate: Learning rate shrinks the contribution of each tree by `learning_rate`. There is a trade-off between learning_rate and n_estimators.
        """
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def __calculate_features(self, data, window=DEFAULT_WINDOW):
        """
        Caculate time features.

        :param data: the time series to detect of
        :param window: the length of window
        """
        features = []
        for index in data:
            if is_standard_time_series(index["data"], window):
                temp = []
                temp.append(extract_features(index["data"], window))
                temp.append(index["flag"])
                features.append(temp)
        return features

    def gbdt_train(self, data, task_id, window=DEFAULT_WINDOW):
        """
        Train a gbdt model.

        :param data: Training dataset.
        :param task_id: The id of the training task.
        :param window: the length of window
        """
        X_train = []
        y_train = []
        features = self.__calculate_features(data, window)
        if features:
            return TSD_LACK_SAMPLE
        for index in features:
            X_train.append(index[0])
            y_train.append(index[1])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        try:
            grd = GradientBoostingClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate)
            grd.fit(X_train, y_train)
            model_name = self.MODEL_PATH + task_id + "_model"
            joblib.dump(grd, model_name)
        except Exception as ex:
            return TSD_TRAIN_ERR, str(ex)
        return TSD_OP_SUCCESS, ""

    def predict(self, X, window=DEFAULT_WINDOW, model_name=DEFAULT_MODEL):
        """
        Predict if a particular sample is an outlier or not.

        :param X: the time series to detect of
        :param type X: pandas.Series
        :param window: the length of window
        :param type window: int
        :param model_name: the model to use
        :param type model_name: string
        :return 1 denotes normal, 0 denotes abnormal
        """
        if is_standard_time_series(X):
            ts_features = extract_features(X, window)
            ts_features = np.array([ts_features])
            load_model = pickle.load(open(model_name, "rb"))
            gbdt_ret = load_model.predict_proba(ts_features)[:, 1]
            if gbdt_ret[0] < self.threshold:
                value = 0
            else:
                value = 1
            return [value, gbdt_ret[0]]
        else:
            return [0, 0]


class XGBoosting(object):
    """
    XGBoost is an optimized distributed gradient boosting library designed to be highly efficient,
    flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.
    XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems
    in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI)
    and can solve problems beyond billions of examples.

    https://github.com/dmlc/xgboost
    """
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/')
    DEFAULT_MODEL = MODEL_PATH + "xgb_default_model"
    def __init__(self,
                 threshold=0.15,
                 max_depth=10,
                 eta=0.05,
                 gamma=0.1,
                 silent=1,
                 min_child_weight=1,
                 subsample=0.8,
                 colsample_bytree=1,
                 booster='gbtree',
                 objective='binary:logistic',
                 eval_metric='auc'):
        """
        :param threshold: The critical point of normal.
        :param max_depth: Maximum tree depth for base learners.
        :param eta: Value means model more robust to overfitting but slower to compute.
        :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree.
        :param silent: If 1, it will print information about performance. If 2, some additional information will be printed out.
        :param min_child_weight: Minimum sum of instance weight(hessian) needed in a child.
        :param subsample: Subsample ratio of the training instance.
        :param colsample_bytree: Subsample ratio of columns when constructing each tree.
        :param booster: Specify which booster to use: gbtree, gblinear or dart.
        :param objective: Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).
        :param eval_metric: If a str, should be a built-in evaluation metric to use. See doc/parameter.md. If callable, a custom evaluation metric.
        """
        self.threshold = threshold
        self.max_depth = max_depth
        self.eta = eta
        self.gamma = gamma
        self.silent = silent
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.booster = booster
        self.objective = objective
        self.eval_metric = eval_metric

    def __save_libsvm_format(self, data, feature_file_name):
        """
        Save the time features to libsvm format.

        :param data: feature values
        :param file_name: file saves the time features and label
        """
        try:
            f = open(feature_file_name, "w")
        except Exception as ex:
            return TSD_CAL_FEATURE_ERR, str(ex)
        times = 0
        for temp in data:
            if times > 0:
                f.write("\n")
            result = ['{0}:{1}'.format(int(index) + 1, value) for index, value in enumerate(temp[0])]
            f.write(str(temp[1]))
            for x in result:
                f.write(' ' + x)
            times = times + 1
        return TSD_OP_SUCCESS, ""

    def __calculate_features(self, data, feature_file_name, window=DEFAULT_WINDOW):
        """
        Caculate time features and save as libsvm format.

        :param data: the time series to detect of
        :param feature_file_name: the file to use
        :param window: the length of window
        """
        features = []
        for index in data:
            if is_standard_time_series(index["data"], window):
                temp = []
                temp.append(extract_features(index["data"], window))
                temp.append(index["flag"])
                features.append(temp)
        try:
            ret_code, ret_data = self.__save_libsvm_format(features, feature_file_name)
        except Exception as ex:
            ret_code = TSD_CAL_FEATURE_ERR
            ret_data = str(ex)
        return ret_code, ret_data

    def xgb_train(self, data, task_id, num_round=300):
        """
        Train an xgboost model.

        :param data: Training dataset.
        :param task_id: The id of the training task.
        :param num_round: Max number of boosting iterations.
        """
        model_name = self.MODEL_PATH + task_id + "_model"
        feature_file_name = self.MODEL_PATH + task_id + "_features"
        ret_code, ret_data = self.__calculate_features(data, feature_file_name)
        if ret_code != TSD_OP_SUCCESS:
            return ret_code, ret_data
        try:
            dtrain = xgb.DMatrix(feature_file_name)
        except Exception as ex:
            return TSD_READ_FEATURE_FAILED, str(ex)
        params = {
            'max_depth': self.max_depth,
            'eta': self.eta,
            'gamma': self.gamma,
            'silent': self.silent,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'booster': self.booster,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
        }
        try:
            bst = xgb.train(params, dtrain, num_round)
            bst.save_model(model_name)
        except Exception as ex:
            return TSD_TRAIN_ERR, str(ex)
        return TSD_OP_SUCCESS, ""

    def predict(self, X, window=DEFAULT_WINDOW, model_name=DEFAULT_MODEL):
        """
        :param X: the time series to detect of
        :type X: pandas.Series
        :param window: the length of window
        :param model_name: Use a xgboost model to predict a particular sample is an outlier or not.
        :return 1 denotes normal, 0 denotes abnormal.
        """
        if is_standard_time_series(X, window):
            ts_features = []
            features = [10]
            features.extend(extract_features(X, window))
            ts_features.append(features)
            res_pred = xgb.DMatrix(np.array(ts_features))
            bst = xgb.Booster({'nthread': 4})
            bst.load_model(model_name)
            xgb_ret = bst.predict(res_pred)
            if xgb_ret[0] < self.threshold:
                value = 0
            else:
                value = 1
            return [value, xgb_ret[0]]
        else:
            return [0, 0]