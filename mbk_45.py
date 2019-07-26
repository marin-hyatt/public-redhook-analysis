#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import PolynomialFeatures
import datetime 
import pytz
from pytz import timezone
import tarfile
from numpy import load
import os
from sklearn.manifold import TSNE
import h5py
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import scipy as sp
from scipy import spatial
from scipy import stats

h5 = h5py.File('sound_data_improved.hdf5', 'r')

d = h5['sound_data']

pca_45 = sklearnPCA(45) 
projected = pca_45.fit_transform(d['feature_vector'])

projected_tsne = TSNE(n_components=2).fit_transform(projected)

# Nearest Neighbors and Corresponding Audio Files

def get_cluster_centers(num_clusters):
    mbk = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    mbk_clusters = mbk.fit_predict(projected)
    return mbk.cluster_centers_

def nearest_neighbors_df(num_clusters):
    cluster_centers_ = get_cluster_centers(num_clusters)
    size = cluster_centers_.shape[0]*5
    neighbors_arr = []
    centroids = []
    centroid_num_arr = []
    num_clusters_arr = size*[num_clusters]
    neighbors_timestamps_orig = []
    neighbors_timestamps = []
    neighbors_sensor_id = []
    neighbors_file_path = []
    neighbors_file_path_cut = []
    neighbors_timestamps_dt = np.empty(size, dtype = datetime.datetime)
    nearest_neighbors = tree.query(cluster_centers_, 5)
    make_neighbors_arr(nearest_neighbors, centroid_num, cluster_centers_, neighbors_arr, centroids, centroid_num_arr)
    retrieve_neighbors_info(neighbors_arr, neighbors_timestamps_orig, neighbors_timestamps, neighbors_sensor_id, neighbors_file_path)
    cut_file_path(neighbors_file_path,  neighbors_file_path_cut)
    convert_timestamp(neighbors_timestamps, size, neighbors_timestamps_dt)
    return make_df(centroids, neighbors_timestamps_orig, neighbors_timestamps_dt, neighbors_sensor_id, \
            neighbors_file_path_cut, centroid_num_arr, num_clusters_arr)

def make_neighbors_arr(nearest_neighbors, centroid_num, cluster_centers_, neighbors_arr, centroids, centroid_num_arr):
    centroid_num = 0
    for x in nearest_neighbors[1]:
        for y in x:
            feature_vector = projected[y]
            neighbors_idx = stats.mode(np.where(projected==feature_vector)[0])
            #Have to do mode because matches in projected_45 covered almost a whole row, but in some cases shifted a little
            neighbors_arr.append(neighbors_idx.mode[0])
            centroids.append(cluster_centers_[centroid_num])
            centroid_num_arr.append(centroid_num+1)
        centroid_num += 1

def retrieve_neighbors_info(neighbors_arr, neighbors_timestamps_orig, neighbors_timestamps, neighbors_sensor_id, neighbors_file_path):
    for f in neighbors_arr:
        neighbors_timestamps_orig.append(d[f, 'timestamp_orig'])
        neighbors_timestamps.append(d[f, 'timestamp'])
        neighbors_sensor_id.append(d[f, 'sensor_id'])
        neighbors_file_path.append(d[f, 'file_path'])

def cut_file_path(neighbors_file_path, neighbors_file_path_cut):
    for path in neighbors_file_path:
        neighbors_file_path_cut.append(path[32:])

def convert_timestamp(neighbors_timestamps, size, neighbors_timestamps_dt):
    for i in range(size):
        j = neighbors_timestamps[i]
        dt = datetime.datetime.utcfromtimestamp(j)
        dt = pytz.UTC.localize(dt)
        dt = dt.astimezone(pytz.timezone('US/Eastern'))
        neighbors_timestamps_dt[i] = dt

def make_df(centroids, neighbors_timestamps_orig, neighbors_timestamps_dt, neighbors_sensor_id, \
            neighbors_file_path_cut, centroid_num_arr, num_clusters_arr):
    df = pd.DataFrame(centroids)
    df.insert(0, "timestamp_orig", neighbors_timestamps_orig, True)
    df.insert(1, "timestamp_dt", neighbors_timestamps_dt, True)
    df.insert(2, "sensor_id", neighbors_sensor_id, True)
    df.insert(3, "file_path", neighbors_file_path_cut, True)
    df.insert(4, "centroid_num", centroid_num_arr, True)
    df.insert(5, "num_clusters", num_clusters_arr, True)
#         all_df.append(df)
    return df

df_2 = nearest_neighbors_df(2)

for n_clusters in range(3, 16):
    df_2 = pd.concat([nearest_neighbors_df(n_clusters), df_2], ignore_index=True)

for pow in range(4,7):
    df_2 = pd.concat([nearest_neighbors_df(2**pow), df_2], ignore_index=True)

df_2.to_csv('mbk_45.csv')