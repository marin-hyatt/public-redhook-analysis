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

#Only include data from middle sensors
sensor_mask = (d['sensor_id'] == b'sonycnode-b827ebc178d2.sonyc')

pca_45 = sklearnPCA(45)
projected = pca_45.fit_transform(d['feature_vector'])

print('done with PCA')

# projected_tsne = TSNE(n_components=2).fit_transform(projected)

# print('done with TSNE')

# Nearest Neighbors and Corresponding Audio Files

def get_cluster_centers(num_clusters):
    """

    Parameters
    ----------
    num_clusters

    Returns
    -------
    mbk.cluster_centers_ : array of shape (num_clusters, 45)
        An array of the feature vectors for each centroid in each cluster.

    """
    mbk = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    mbk_clusters = mbk.fit_predict(projected[sensor_mask])
    return mbk.cluster_centers_


def cut_file_path(neighbors_file_path, neighbors_file_path_cut):
    """
    Cuts a file name to start with the sensor name.
    """
    for path in neighbors_file_path:
        neighbors_file_path_cut.append(path[32:])
        

def make_neighbors_dataframe(num_clusters):
    """
    Returns a DataFrame with the information (timestamp, filepath, etc) from five neighbors of each centroid in each
    cluster the data is grouped into.

    Parameters
    ----------
    num_clusters : int
        Number of clusters to group the projected data into.

    Returns
    -------
    df : DataFrame
        pandas DataFrame listing information about each neighbor, including timestamp, filepath, the centroid it is
        associated with, and the number of clusters the projected data is grouped into.
    """
    tree = spatial.KDTree(projected[sensor_mask])

    cluster_centers = get_cluster_centers(num_clusters)

    nearest_neighbors = tree.query(cluster_centers, 5)

    # Creates array of indices of elements in projected_45 that match the neighbors
    # Creating array of centroid feature vectors corresponding to each neighbor, also which centroid the neighbor belongs to

    centroids = []
    neighbors_arr = []
    centroid_num_arr = []
    for centroid_num, x in enumerate(nearest_neighbors[1]):
        for y in x:
            neighbors_idx = np.nonzero(sensor_mask)[0][y]
            neighbors_arr.append(neighbors_idx)
            centroids.append(cluster_centers[centroid_num])
            centroid_num_arr.append(centroid_num + 1)

    # Mask for elements of d that are neighbors
    index_2 = np.zeros(d.shape[0]).astype('bool')
    index_2[np.sort(neighbors_arr)] = True

    # Creating array with number of clusters for each entry
    num_clusters_arr = len(neighbors_arr) * [num_clusters]

    # Converting timestamps to datetime format
    neighbors_timestamps_dt = []
    for for i in neighbors_arr:
        j = d[i, 'timestamp']
        dt = datetime.datetime.utcfromtimestamp(j)
        dt = pytz.UTC.localize(dt)
        dt = dt.astimezone(pytz.timezone('US/Eastern'))
        neighbors_timestamps_dt.append(dt)

    # Cutting the filepath so it starts with the sensor name
    test_cut_path = []
    cut_file_path(d[index_2]['file_path'], test_cut_path)

    # Making the dataframe
    df = pd.DataFrame(centroids)
    df.insert(0, "timestamp_orig", d[index_2]['timestamp_orig'], True)
    df.insert(1, "timestamp_dt", neighbors_timestamps_dt, True)
    df.insert(2, "sensor_id", d[index_2]['sensor_id'], True)
    df.insert(3, "file_path", test_cut_path, True)
    df.insert(4, "centroid_num", centroid_num_arr, True)
    df.insert(5, "num_clusters", num_clusters_arr, True)

    return df

print('done with defining functions')

# Creates one large DataFrame with nearest neighbors information for 2 to 15 clusters, then also 16, 32, and 64 clusters

df_2 = make_neighbors_dataframe(2)
df_2.to_csv('mbk_45_middle_sensors.csv')

print('done with creating dataframe for 2 clusters')

for n_clusters in range(3, 16):
    df_2 = pd.concat([make_neighbors_dataframe(n_clusters), df_2], ignore_index=True)
    df_2.to_csv('mbk_45_middle_sensors.csv')
    print('done with creating dataframe for ' + str(n_clusters) + ' clusters')
    


for power in range(4, 7):
    df_2 = pd.concat([make_neighbors_dataframe(2 ** power), df_2], ignore_index=True)
    df_2.to_csv('mbk_45_middle_sensors.csv')
    print('done with creating dataframe for ' + str(2 ** power) + ' clusters')

df_2.to_csv('mbk_45_middle_sensors.csv')
