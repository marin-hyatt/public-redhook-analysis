#!/usr/bin/env python3
print('start')
import sys
sys.stdout.flush()
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

print('done with importing')
sys.stdout.flush()

h5 = h5py.File('sound_data_improved.hdf5', 'r')

d = h5['sound_data']

middle_sensor_mask = (d['sensor_id'] == b'sonycnode-b827ebc178d2.sonyc') | (d['sensor_id'] == b'sonycnode-b827eb491436.sonyc')

d_middle = d[middle_sensor_mask]

pca_106 = sklearnPCA(106)
projected = pca_106.fit_transform(d_middle['feature_vector'])

print('done with pca')
sys.stdout.flush()

def get_cluster_model(num_clusters):
    """
    Returns the MiniBatchKMeans model fitted to the projected dataset.
    
    Parameters
    ----------
    num_clusters : int
        The number of clusters to group the data into.

    Returns
    -------
    mbk : MiniBatchKMeans object
        The fitted model.

    """
    mbk = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    mbk.fit(projected)
    return mbk

def cut_file_path(neighbors_file_path):
    """
    Cuts a file name to start with the sensor name.
    
    Parameters
    ----------
    neighbors_file_path : string
        String representing the file path.
        
    Returns
    -------
    neighbors_file_path[32:] : string
        File path starting with the sensor name.
    """
    return(neighbors_file_path[32:])   

def convert_to_dt(timestamp):
    """
    Converts a float timestamp to a datetime object.
    
    Parameters
    ----------
    timestamp : float
        A float representing the time.
        
    Returns
    -------
    dt : datetime object
        A datetime object corresponding to the time represented by timestamp.
    """
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    dt = pytz.UTC.localize(dt)
    dt = dt.astimezone(pytz.timezone('US/Eastern'))
    return dt
    
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
    cluster_assignments = get_cluster_model(num_clusters).predict(projected)
    cluster_centers = get_cluster_model(num_clusters).cluster_centers_
    centroid_cluster_assignments = get_cluster_model(num_clusters).predict(cluster_centers)

    # print(centroid_cluster_assignments)
    # print(len(cluster_centers))

    centroids = []
    centroid_num_arr = []
    num_centroids = num_clusters*10*[num_clusters]
    
    is_neighbor = []
    
    #arrays for neighbors
    timestamps = np.empty((num_clusters,10))
    timestamps_orig = np.empty((num_clusters,10))
    file_path = np.empty((num_clusters,10), dtype='S92')
    # neighbor_file_path = []
    sensor_id = np.empty((num_clusters,10), dtype='S60')
    # neighbor_timestamps_dt = np.empty((64*5), dtype = datetime.datetime)
    # print(neighbor_timestamps_dt.dtype)

    for i,cluster_index in enumerate(range(num_clusters)):
        #for each cluster center, query only the cluster it belongs to

        #Filter out only the points belonging to one cluster
        cluster_mask = (cluster_assignments==cluster_index)
        cluster_test = projected[cluster_mask]

        #Makes a list of the centroid of the cluster with length of the number of the points in the cluster
        centroid_list = 10*[cluster_centers[cluster_index]]
        centroids += centroid_list

        #Makes a list of the cluster index with length of the number of the points in the cluster
        centroid_num_list = 10*[cluster_index+1]
        centroid_num_arr += centroid_num_list

#         print(len(cluster_test))
        nearest_neighbors = []
        tree = spatial.KDTree(cluster_test)
    #     print(cluster_centers[cluster_index])
        nearest_neighbors = tree.query(cluster_centers[cluster_index], 5)[1]

        #from only the points corresponding to a certain cluster in the 10000 subset of projected, apply the nearest
        #neighbors mask to obtain the other characteristics like file path, timestamp, etc

        neighbors_mask = np.zeros(len(cluster_test)).astype('bool')
        neighbors_mask[np.sort(nearest_neighbors)] = True
        is_neighbor += 5*['Y']
        
         #random sampling from cluster 
        random_nums = np.random.choice(range(cluster_test.shape[0]), 5, replace=False)
        random_cluster_mask = np.zeros(cluster_test.shape[0]).astype('bool')
        random_cluster_mask[random_nums] = True
        is_neighbor += 5*['N']
        
        
        d_neighbors = d_middle[cluster_mask][neighbors_mask]
        d_random = d_middle[cluster_mask][random_cluster_mask]
        
        timestamps_empty = np.empty((2, 5))
        timestamps_empty[0] = d_neighbors['timestamp']
        timestamps_empty[1] = d_random['timestamp']
        timestamps[i] = (timestamps_empty.flatten())
        
        timestamps_orig_empty = np.empty((2, 5))
        timestamps_orig_empty[0] = d_neighbors['timestamp_orig']
        timestamps_orig_empty[1] = d_random['timestamp_orig']
        timestamps_orig[i] = timestamps_orig_empty.flatten()
        
        file_path_empty = np.empty((2, 5), dtype='S92')
        file_path_empty[0] = d_neighbors['file_path']
        file_path_empty[1] = d_random['file_path']
    #     print(neighbor_file_path_inner)
        file_path[i] = file_path_empty.flatten()
        
        sensor_id_empty = np.empty((2, 5), dtype='S60')
        sensor_id_empty[0] = d_neighbors['sensor_id']
        sensor_id_empty[1] = d_random['sensor_id']
        sensor_id[i] = sensor_id_empty.flatten()
        
#         print('done with cluster ' + str(cluster_index) + ' of ' + str(num_clusters))
#         sys.stdout.flush()

    timestamps_dt = [convert_to_dt(x) for x in timestamps.flatten()]
    file_path_cut = [cut_file_path(x) for x in file_path.flatten()]
    
#     print(len(is_neighbor))
    
    # Making the dataframe
    df = pd.DataFrame(centroids)
    df.insert(0, 'is_neighbor', is_neighbor, True)
    df.insert(1, "timestamp_orig", timestamps_orig.flatten(), True)
    df.insert(2, "timestamp_dt", timestamps_dt, True)
    df.insert(3, "sensor_id", sensor_id.flatten(), True)
    df.insert(4, "file_path", file_path_cut, True)
    df.insert(5, "centroid_num", centroid_num_arr, True)
    df.insert(6, "num_clusters", num_centroids, True)

    return df


print('done with defining functions')
sys.stdout.flush()

df_2 = make_neighbors_dataframe(2)
df_2.to_csv('mbk_106_3.csv')

print('done with creating dataframe for 2 clusters')
sys.stdout.flush()

for n_clusters in range(3, 16):
    df_2 = pd.concat([make_neighbors_dataframe(n_clusters), df_2], ignore_index=True)
    df_2.to_csv('mbk_106_3.csv')
    print('done with creating dataframe for' + str(n_clusters) + 'clusters')
    sys.stdout.flush()
    


for power in range(4, 8):
    df_2 = pd.concat([make_neighbors_dataframe(2 ** power), df_2], ignore_index=True)
    df_2.to_csv('mbk_106_3.csv')
    print('done with creating dataframe for' + str(2**power) + 'clusters')
    sys.stdout.flush()

df_2.to_csv('mbk_106_3.csv')



