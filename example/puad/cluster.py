""" Rocka model for time series clustering"""
import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir)

import numpy as np
import pandas as pd
from example.puad.rocka import Rocka, SBD, density_radius
from example import ADdataset, Preprocessor
from collections import defaultdict


def get_the_medoids(sim_matrix, indexes):
    """
    calculate the center curve of each cluster, whose sum(square(sim to other curves in the same cluster)) is the smallest.

    Params
    ------
    sim_matrix: 2-D ndarray, similarity matrix of kpi curves.
    indexes: list, the indexes of kpi curves with same cluster.

    Returns
    -------
    mediod: int, the index of the center curve.
    min_dist: float, the minimum sum distance.
    """
    min_dist = sys.maxsize
    medoid = -1
    for start in indexes:
        dist = 0
        for end in indexes:
            if end != start:
                dist += np.square(sim_matrix[start][end])
        if dist < min_dist:
            min_dist = dist
            medoid = start
    return medoid, min_dist

def assign_to_nearest(sim_matrix, noise_index, labels, max_dis):
    """
    assign the noise in dbscan to the cluster which its nearest neighbor belongs to.

    Params
    ------
    sim_matrix: 2-D ndarray, similarity matrix of kpi curves.
    noise_index: int, the index of noise kpi.
    labels: 1-D ndarray, the clustering labels.
    max_dis: float, the maximum similarity threshold.

    Returns
    -------
    int: the cluster id.
    1-D ndarray, new clustering labels.
    """
    sim_dis = sim_matrix[noise_index][:]
    sim_dis_copy = sim_dis.tolist()
    sim_dis.sort()
    for i in sim_dis:
        if i < max_dis:
            index = sim_dis_copy.index(i)
            if labels[index] != -1:
                labels[noise_index] = labels[index]
                return labels[noise_index], labels
            else:
                continue
        else:
            labels[noise_index] = -1
            return labels[noise_index], labels
    return labels[noise_index], labels

def run_dbscan(sim_matrix, radius, minPts, kpilist):
    """
    Run dbscan clustering algorithm, find the center kpi for each cluster, and assign noisy curve to its nearest cluster.

    Params
    ------
    sim_matrix: 2-D ndarray, similarity matrix of kpi curves.
    radius: float, the maximum neighbourhood distance in DBSCAN.
    minPts: int, the minimum neighbourhood samples in DBSCAN.
    kpilist: list, list of kpis.

    Returns
    -------
    medoids: list, the center kpi of each cluster.
    labels_cal: 1-D ndarray, the clustering labels.
    """
    # dbscan model
    model = Rocka(density_radius=radius, minPts=minPts).fit(sim_matrix)
    # core_sample_mask = np.zeros_like(model.labels_, dtype=bool)
    # core_sample_mask[model.core_sample_indices_] = True
    labels_cal = model.labels_
    # number of clusters, ignoring noise if present
    num_clusters = len(set(labels_cal)) - (1 if -1 in labels_cal else 0)
    medoids = []

    for cla in range(num_clusters):
        index = [idx for idx, e in enumerate(labels_cal) if e == cla]
        medoid, min_dist = get_the_medoids(sim_matrix, index)
        medoids.append(kpilist[medoid])
        # print("cluster: {}, mediod kpi: {}, kpi number: {}".format(cla, kpilist[medoid], len(index)))

    # assign the 'noisy' curve in DBSCAN according to its nearest clustered curve.
    index = [idx for idx, e in enumerate(labels_cal) if e == -1]
    for uuid in index:
        cla, new_labels = assign_to_nearest(sim_matrix, uuid, labels_cal, radius*1.2)
        if cla == -1:
            # print("assign failed for noisy kpi {}, sim distance: {}".format(kpilist[uuid], sim_matrix[uuid][:]))
            labels_cal[uuid] = len(medoids)
            medoids.append(kpilist[uuid])
        else:
            labels_cal = new_labels
    return medoids, labels_cal

def run_rocka(datadir="data", dataset="KPI", minPts=4):
    """
    run the rocka model for kpi clustering.

    Params
    ------
    datadir: str, the data folder name.
    dataset: str, the dataset name.
    minPts: int, the minimum neighbourhood samples in DBSCAN.

    Returns
    -------
    df: DataFrame, the clustering result of kpis and the center kpi for each cluster. 
    """
    # 读取数据集
    data_dir = os.path.join(project_dir, datadir)
    dataset = ADdataset(root=data_dir, dataset=dataset)
    kpis = defaultdict(list)
    values = defaultdict(list)
    for name, (train_df, train_label), (test_df, test_label) in dataset:
        # 预处理
        train_pre = Preprocessor(train_df, train_label, normalize=True, fillna="interpolation", smooth=0.05)
        train_df, train_label, train_missing = train_pre.process()
        new_values, residuals = train_pre.extract_baseline(train_df["value"].values)
        time_interval = int(train_df["timestamp"].diff().min())
        kpis[time_interval].append(name)
        values[time_interval].append(new_values)

    radius_dict = {60: 0.45, 300: 0.35}
    kpilist = []
    labellist = []
    centroidlist = []
    # 根据时间间隔将kpi分组
    for interval, valuelist in values.items():
        sbd_matrix, ret_sbd = SBD(valuelist, minPts=minPts)
        # print(sbd_matrix)
        # print(ret_sbd)
        # radius = density_radius(sbd_arr=ret_sbd, len_thresh=4, max_radius=1, slope_thresh=0.1, slope_diff_thresh=0.001)
        # print(radius)
        radius = radius_dict[interval]
        medoids, labels = run_dbscan(sbd_matrix, radius, minPts, kpis[interval])
        if kpilist:
            labellist += list(labels + len(set(labellist)))
        else:
            labellist += list(labels)
        kpilist += kpis[interval]
        centroidlist += medoids
    # 保存结果
    df = pd.DataFrame()
    df["kpi"] = kpilist
    df["cluster"] = labellist
    df["centroid"] = df[["kpi", "cluster"]].apply(lambda x: x[1] if x[0] in centroidlist else -1, axis=1)
    return df
    
if __name__ == "__main__":
    df = run_rocka(minPts=3)
    df.to_csv("rocka.csv", index=False)