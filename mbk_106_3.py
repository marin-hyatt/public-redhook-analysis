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

pca_106 = sklearnPCA(106)
projected = pca_106.fit_transform(d['feature_vector'])

print('done with pca')

def get_cluster_model(num_clusters):
    """

    Parameters
    ----------
    num_clusters

    Returns
    -------
    mbk.cluster_centers_ : array of shape (num_clusters, 106)
        An array of the feature vectors for each centroid in each cluster.

    """
    mbk = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    mbk.fit(projected)
    return mbk

def cut_file_path(neighbors_file_path):
    """
    Cuts a file name to start with the sensor name.
    """
    neighbors_file_path_cut = []
    for path in neighbors_file_path:
        neighbors_file_path_cut.append(path[32:])   
        
    return neighbors_file_path_cut

def convert_to_dt(timestamp):
        dt = datetime.datetime.utcfromtimestamp(timestamp)
        dt = pytz.UTC.localize(dt)
        dt = dt.astimezone(pytz.timezone('US/Eastern'))
        return dt
    
def make_neighbors_dataframe(num_clusters):
    cluster_assignments = get_cluster_model(num_clusters).predict(projected)
    cluster_centers = get_cluster_model(num_clusters).cluster_centers_
    centroid_cluster_assignments = get_cluster_model(num_clusters).predict(cluster_centers)

    centroids = []
    centroid_num_arr = []
    num_centroids = num_clusters*[num_clusters]
    neighbor_timestamps = np.empty((num_clusters,5))
    neighbor_timestamps_orig = np.empty((num_clusters,5))
    neighbor_cut_file_path = np.empty((num_clusters,5), dtype='S60')
    neighbor_sensor_id = np.empty((num_clusters,5), dtype='S60')
   
    for i,cluster_index in enumerate(range(num_clusters)):
        #for each cluster center, query only the cluster it belongs to

        #Filter out only the points belonging to one cluster
        cluster_mask = (cluster_assignments==cluster_index)
        cluster_test = projected[cluster_mask]

        #Makes a list of the centroid of the cluster with length of the number of the points in the cluster
        centroid_list = 5*[cluster_centers[cluster_index]]
        centroids += centroid_list

        #Makes a list of the cluster index with length of the number of the points in the cluster
        centroid_num_list = 5*[cluster_index]
        centroid_num_arr += centroid_num_list

        print(len(cluster_test))
        nearest_neighbors = []
        tree = spatial.KDTree(cluster_test)
        nearest_neighbors = tree.query(cluster_centers[cluster_index], 5)[1]

        #from only the points corresponding to a certain cluster in the 10000 subset of projected, apply the nearest
        #neighbors mask to obtain the other characteristics like file path, timestamp, etc

        neighbors_mask = np.zeros(len(cluster_test)).astype('bool')
        neighbors_mask[np.sort(nearest_neighbors)] = True

        neighbor_timestamps_prelim = d[cluster_mask][neighbors_mask]['timestamp']
    
        neighbor_timestamps[i] = (neighbor_timestamps_prelim)
        neighbor_timestamps_orig[i] = (d[cluster_mask][neighbors_mask]['timestamp_orig'])
        neighbor_cut_file_path[i] = (cut_file_path(d[cluster_mask][neighbors_mask]['file_path']))
        neighbor_sensor_id[i] = (d[cluster_mask][neighbors_mask]['sensor_id'])
        neighbor_timestamps_dt = [convert_to_dt(x) for x in neighbor_timestamps.flatten()]
       
    df = pd.DataFrame(centroids)
    df.insert(0, "timestamp_orig", neighbor_timestamps_orig.flatten(), True)
    df.insert(1, "timestamp_dt", neighbor_timestamps_dt, True)
    df.insert(2, "sensor_id", neighbor_sensor_id.flatten(), True)
    df.insert(3, "file_path", neighbor_cut_file_path.flatten(), True)
    df.insert(4, "centroid_num", centroid_num_arr, True)
    df.insert(5, "num_clusters", num_centroids, True)
    
    return df

print('done with defining functions')

df_2 = make_neighbors_dataframe(2)
df_2.to_csv('mbk_106_3.csv')

print('done with creating dataframe for 2 clusters')

for n_clusters in range(3, 16):
    df_2 = pd.concat([make_neighbors_dataframe(n_clusters), df_2], ignore_index=True)
    df_2.to_csv('mbk_106_3.csv')
    print('done with creating dataframe for' + str(n_clusters) + 'clusters')
    


for power in range(4, 7):
    df_2 = pd.concat([make_neighbors_dataframe(2 ** power), df_2], ignore_index=True)
    df_2.to_csv('mbk_106_3.csv')
    print('done with creating dataframe for' + str(power) + 'clusters')

df_2.to_csv('mbk_106_3.csv')



