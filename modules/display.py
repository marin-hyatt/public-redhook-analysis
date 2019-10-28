import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from pytz import timezone
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import librosa
import scipy
from scipy import ndimage
from scipy import signal
import matplotlib.dates as md

def print_scipy():
    print(scipy)

def get_cluster_assignments(num_clusters, sensor_transformed, fit_arr):
    """
    Returns an array containing the number of each cluster each data point in sensor_transformed is assigned to.
    Clustering is performed using MiniBatchKMeans.
    
    Parameters
    ----------
    num_clusters : int
        The number of clusters to group sensor_transformed into.
        
    sensor_transformed : array of floats
        45-dimensional array of feature vectors from one sensor 
        
    fit_arr : array of floats
        45-dimensional array of feature vectors from all sensors
        
    Returns
    -------
    cluster_indices : int array
        An array with the index of the cluster that each data point belongs to.
        
    """
    mbk = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    mbk.fit(fit_arr)
    cluster_indices = mbk.predict(sensor_transformed)
    return cluster_indices

def get_y_and_bins(num_clusters, sensor_transformed, fit_arr, sensor_timestamps_dt, default_bins, clusters_plot_arr):
    """
    Returns y values and bin values for a histogram of data. Used to determine the bin values for plotting the 
    histogram.
    
    Parameters
    ----------
    num_clusters : int
        The number of clusters to assign the data to.
        
    sensor_transformed : 45-dimensional array of floats
        An array taken from projected_45 with only the feature vectors corresponding to one sensor.
        
    fit_arr : array of floats
        45-dimensional array of feature vectors from all sensors
        
    sensor_timestamps_dt : datetime array
        Numpy array with dtype=datetime.datetime, containing day values for each data point gathered for a specified
        sensor.
        
    default_bins : int
        Number of bins to group the data into at first.
        
    clusters_plot_arr : int array
        The list of cluster indices to plot.
        
    Returns
    -------
    y : float array
        Array of y values for the histogram
        
    bins : float array
        Array of bin values, to input in the actual histogram plot so it accurately reflects units of time (e.g. days)
        
    """
    
    test = get_cluster_assignments(num_clusters, sensor_transformed, fit_arr)

    cluster_mask = np.nonzero(test==clusters_plot_arr[0]) #boolean mask for specific cluster number
        
    #Makes array of all timestamps where cluster assignment occurred
    timestamp_arr = np.asarray(sensor_timestamps_dt)[cluster_mask]
        
    #Get y values, bins
    y, bins, _ = plt.hist(timestamp_arr, bins=default_bins)
    
    return y, bins

def plot_clusters(num_clusters, sensor_transformed, fit_arr, sensor_timestamps_dt, spl_time, spl_dBAS_mean, \
                  spl_dBAS_max, spl_dBAS_median, num_bins, clusters_plot_arr):
    """
    Plots a histogram of the frequency of cluster assignments over time for one sensor.
    
    Parameters
    ----------
    num_clusters : int
        The number of clusters to assign the data to.
        
    sensor_transformed : 45-dimensional array of floats
        An array taken from projected_45 with only the feature vectors corresponding to one sensor.
        
    fit_arr : array of floats
        45-dimensional array of feature vectors from all sensors
        
    sensor_timestamps_dt : datetime array
        Numpy array with dtype=datetime.datetime, containing day values for each data point gathered for a specified
        sensor.
        
    spl_time : datetime array
        Array of datetime objects corresponding to the SPL values.
        
    spl_dBAS_mean : array of floats
        Array of SPL values corresponding to spl_time, averaged over each minute.
        
    spl_dBAS_max : array of floats
        Array of SPL values corresponding to spl_time, consisting of the maximum value from each minute.
        
    spl_dBAS_median : array of floats
        Array of SPL values corresponding to spl_time, consisting of the median value from each minute.
    
    num_bins : int
        The number of bins to group the cluster frequency into.
        
    bin_arr : array of floats
        Array of bin edges to group the cluster frequency into.
        
    clusters_plot_arr : arr of ints within the range(0, num_clusters)
        The indices of the clusters to plot.
    """
    
    #Setting the figure size, layout
    fig = plt.figure(figsize=(15,100), dpi=60)
    subplot_idx = 1
    y_vals = [] #array of all y values, used to get ylim for graph

    test = get_cluster_assignments(num_clusters, sensor_transformed, fit_arr)

    for cluster_num in clusters_plot_arr:
        cluster_mask = np.nonzero(test==cluster_num) #boolean mask for specific cluster number
        
        #If there are no instances of this cluster in the cluster assignments, don't graph
        if(np.sum(cluster_mask) == 0):
            continue
        
        
        #Makes array of all timestamps where cluster assignment occurred
        timestamp_arr = np.asarray(sensor_timestamps_dt)[cluster_mask]
        
        ax1 = fig.add_subplot(num_clusters, 1, subplot_idx)  
        color = 'tab:red'
        ax1.set_xlabel('time')
        ax1.set_ylabel('cluster ' + str(cluster_num), color=color)
    
        #Get y values, bins
        if(cluster_num == clusters_plot_arr[0]):
            y, bins, _ = ax1.hist(timestamp_arr, bins=num_bins, color=color)

        ax1.hist(timestamp_arr, bins=bins, color=color)
        y_vals.append(y.max())
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('dBAS', color=color)  # we already handled the x-label with ax1
        ax2.plot(spl_time, spl_dBAS_mean, color=color)
        ax2.plot(spl_time, spl_dBAS_max, color='black')
        ax2.plot(spl_time, spl_dBAS_median, color='green')
        ax2.tick_params(axis='y', labelcolor=color)

        subplot_idx += 1 
        ax1.set_ylim([0,max(y_vals)])
    
    plt.tight_layout()
    plt.show()
    return bins

def plot_truck_clusters_all_measures(num_clusters, sensor_transformed, fit_arr, sensor_timestamps_dt, spl_time, spl_dBAS_mean, \
                  spl_dBAS_max, spl_dBAS_median, y_vals, bin_arr, clusters_plot_arr):
    """
    Plots a histogram of the frequency of cluster assignments over time for one sensor.
    
    Parameters
    ----------
    num_clusters : int
        The number of clusters to assign the data to.
        
    sensor_transformed : 45-dimensional array of floats
        An array taken from projected_45 with only the feature vectors corresponding to one sensor.
        
    fit_arr : array of floats
        45-dimensional array of feature vectors from all sensors
        
    sensor_timestamps_dt : datetime array
        Numpy array with dtype=datetime.datetime, containing day values for each data point gathered for a specified
        sensor.
        
    spl_time : datetime array
        Array of datetime objects corresponding to the SPL values.
        
    spl_dBAS_mean : array of floats
        Array of SPL values corresponding to spl_time, averaged over each minute.
        
    spl_dBAS_max : array of floats
        Array of SPL values corresponding to spl_time, consisting of the maximum value from each minute.
        
    spl_dBAS_median : array of floats
        Array of SPL values corresponding to spl_time, consisting of the median value from each minute.
        
    y_vals : array of floats
        array of y values used to calculate the ylim.
        
    bin_arr : array of floats
        Array of bin edges to group the cluster frequency into.
        
    clusters_plot_arr : arr of ints within the range(0, num_clusters)
        The indices of the clusters to plot.
    """
    test = get_cluster_assignments(num_clusters, sensor_transformed, fit_arr)
    
    total_timestamp_arr = []
    y_maxes = [y_vals.max()] #list of all maxes of y values for each cluster number
    for cluster_num in clusters_plot_arr:
        cluster_mask = np.nonzero(test==cluster_num)
        
        if(np.sum(cluster_mask) == 0):
            continue
            
        timestamp_arr = np.asarray(sensor_timestamps_dt)[cluster_mask]
        for timestamp in timestamp_arr:
            total_timestamp_arr.append(timestamp)
    
    fig, ax1 = plt.subplots()  
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('truck clusters', color=color)
        
    y, x, _ = ax1.hist(total_timestamp_arr, bins=bin_arr, color=color)
    y_maxes.append(y.max()) #might need to fix later
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('dBAS', color=color)  # we already handled the x-label with ax1
    ax2.plot(spl_time, spl_dBAS_mean, color=color)
    ax2.plot(spl_time, spl_dBAS_max, color='black')
    ax2.plot(spl_time, spl_dBAS_median, color='green')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_ylim([0,max(y_maxes)])
    
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()
    
def plot_truck_clusters(joined_df, peak_window_size, smoothing_window_size, ds_factor, smoothing):
    """
    Plots the SPL over time with visual cues indicating the presence of trucks over time for one sensor. The shaded
    regions indicate time when sound was recorded (versus just SPL, which is constantly recorded). SPL peaks 
    corresponding to truck activity are plotted with red dots. SPL peaks corresponding to other noises are 
    plotted with gray dots. The SPL is plotted in green. The recorded sound corresponding to trucks is plotted in 
    red. The recorded sound corresponding to anything other than a truck is plotted in light gray.
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking
        
    ds_factor : int
        Downsample factor for getting the median
        
    smoothing : String
        Smoothing type, e.g. mean, median, or gaussian.
    """
    y_vals = []
    joined_df_reset_index = joined_df.reset_index()
    
    #Peaks for all SPL values
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    dBAS_peaks = joined_df['dBAS'][spl_peaks]
    timestamp_peaks = joined_df_reset_index['index'][spl_peaks]
   
    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    print(truck_timestamp_peaks)
    truck_dBAS_peaks = spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1]
    print(truck_dBAS_peaks)
    
    
    #Peaks for SPL values corresponding to other clusters
    other_dBAS_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['dBAS']
    other_timestamp_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['index']
    
    fig, ax1 = plt.subplots()  
    color = 'b'
    
    ax1.set_xlabel('time')
    ax1.set_ylabel('SPL', color=color)
        
    #Plot all SPL peaks
    #Plot dot over max SPL value when that value corresponds to a trucking instance
    ax1.scatter(truck_timestamp_peaks, truck_dBAS_peaks, color='r', s=20)
    ax1.tick_params(axis='y', labelcolor=color)
    
    #Plot SPL peaks corresponding to other clusters
    ax1.scatter(other_timestamp_peaks, other_dBAS_peaks, color='tab:gray', s=20)
    ax1.set_xlim([joined_df_reset_index['index'].iloc[0], joined_df_reset_index['index'].iloc[-1]])
    
    if smoothing == 'median':
        medfit = scipy.signal.medfilt(joined_df['dBAS'].values, smoothing_window_size)[::ds_factor]
        ax1.plot(joined_df.iloc[::ds_factor].reset_index()['index'], medfit, color='g')
        max_y = medfit
    elif smoothing == 'mean':
        mean_filter = scipy.ndimage.convolve(joined_df['dBAS'].values, 
                                             np.ones(smoothing_window_size) / smoothing_window_size)[::ds_factor]
        ax1.plot(joined_df.iloc[::ds_factor].reset_index()['index'], mean_filter, color='g')
        max_y = mean_filter        
    elif smoothing == 'gaussian':
        gaussian_filter = scipy.ndimage.filters.gaussian_filter(joined_df['dBAS'].values, smoothing_window_size)[::ds_factor]
        ax1.plot(joined_df.iloc[::ds_factor].reset_index()['index'], gaussian_filter, color='g')
        max_y = gaussian_filter
    else:
        raise Exception('Unknown smoothing type')
    
    print(len(joined_df.iloc[::ds_factor]))
    #Plot regions corresponding to trucks
    ax1.fill_between(x=joined_df.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(joined_df.iloc[::ds_factor]['dBAS'])*[joined_df['dBAS'].min()], \
                     y2=max_y, \
                     where=joined_df.iloc[::ds_factor].reset_index()['assignment']==1, \
                     color='r', alpha=0.5)
    
    #Plot regions corresponding to other clusters
    ax1.fill_between(x=joined_df.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(joined_df.iloc[::ds_factor]['dBAS'])*[joined_df['dBAS'].min()], \
                     y2=max_y, where=joined_df.iloc[::ds_factor].reset_index()['assignment']==2, \
                     color='tab:gray', alpha=0.5)
    
    ax1.set_ylim(joined_df['dBAS'].min())
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()
    
def plot_truck_clusters_median(joined_df_median, peak_window_size, \
                                       smoothing_window_size, smoothing_window_size_ambient, ds_factor, smoothing):
    """
    Plots the SPL over time with visual cues indicating the presence of trucks over time for one sensor. The shaded
    regions indicate time when sound was recorded (versus just SPL, which is constantly recorded). 
    SPL peaks corresponding to truck activity are plotted with red dots. SPL peaks corresponding to other noises are 
    plotted with gray dots. The median SPL is plotted in blue. The SPL is plotted in green. The recorded sound 
    corresponding to trucks is plotted in red. The recorded sound corresponding to anything other than a truck is 
    plotted in light gray.
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking. Cannot be lower than 3.
        
    smoothing_window : float
        Parameter for smoothing the current SPL. Increasing it smoothes the curve more.
        
    smoothing_window_size_ambient : int
        Parameter for smoothing the ambient SPL. Increasing it smoothes the curve more.
        
    ds_factor : int
        Downsample factor for getting the median
        
    'smoothing' : String
        Type of smoothing. Either median, mean, or gaussian.
    """
    y_vals = []
    joined_df_reset_index = joined_df_median.reset_index()

    #Peaks for all SPL values
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df_median['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    dBAS_peaks = joined_df_median['dBAS'][spl_peaks]
    timestamp_peaks = joined_df_reset_index['index'][spl_peaks]
   
    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    truck_dBAS_peaks = spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1]
    
    #Peaks for SPL values corresponding to other clusters
    other_dBAS_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['dBAS']
    other_timestamp_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['index']
    
    fig, ax1 = plt.subplots()  
    color = 'b'
    
    ax1.set_xlabel('time')
    ax1.set_ylabel('SPL', color=color)
        
    #Plot dot over max SPL value when that value corresponds to a trucking instance
    ax1.scatter(truck_timestamp_peaks, truck_dBAS_peaks, color='r', s=20)
    ax1.tick_params(axis='y', labelcolor=color)
    
    #Plot SPL peaks corresponding to other clusters
    ax1.scatter(other_timestamp_peaks, other_dBAS_peaks, color='tab:gray', s=20)
    ax1.set_xlim([joined_df_reset_index['index'].iloc[0], joined_df_reset_index['index'].iloc[-1]])
    
    if smoothing == 'median':
        plot_y = scipy.signal.medfilt(joined_df_median['dBAS'].values, smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.signal.medfilt(joined_df_median['median_dBAS'].values, smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'mean':
        plot_y = scipy.ndimage.convolve(joined_df_median['dBAS'].values, 
                                             np.ones(smoothing_window_size) / smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.convolve(joined_df_median['median_dBAS'].values, \
                                                    np.ones(smoothing_window_size_ambient) / smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'gaussian':
        plot_y = scipy.ndimage.filters.gaussian_filter(joined_df_median['dBAS'].values, \
                                                                smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.filters.gaussian_filter(joined_df_median['median_dBAS'].values, \
                                                                smoothing_window_size_ambient)[::ds_factor]
    else:
        raise Exception('Unknown smoothing type')
        
    #Plot SPL and median SPL
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], plot_y, color='g')
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], plot_y_median, color='b')
    
    #Shade regions of sound that correspond to trucks according to cluster assignment
    ax1.fill_between(x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(joined_df_median.iloc[::ds_factor]['dBAS'])*[joined_df_median['dBAS'].min()], \
                     y2=plot_y, \
                     where=joined_df_median.iloc[::ds_factor]['assignment']==1, \
                     color='r', alpha=0.5)
    
    #Plot regions corresponding to other clusters
    ax1.fill_between(x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(joined_df_median.iloc[::ds_factor]['dBAS'])*[joined_df_median['dBAS'].min()], \
                     y2=plot_y, where=joined_df_median.iloc[::ds_factor]['assignment']==2, \
                     color='tab:gray', alpha=0.5)
    
    ax1.set_ylim(joined_df_median['dBAS'].min())
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()
                                               
def plot_truck_clusters_median_shading(joined_df_median, peak_window_size, \
                                       smoothing_window_size, smoothing_window_size_ambient, ds_factor, smoothing):
    """
    Plots the SPL over time with visual cues indicating the presence of trucks over time for one sensor. The shaded
    regions indicate time when sound was recorded (versus just SPL, which is constantly recorded). There is only
    shading when the SPL recorded at the time is greater than the median SPL. 
    SPL peaks corresponding to truck activity are plotted with red dots. SPL peaks corresponding to other noises are 
    plotted with gray dots. The median SPL is plotted in blue. The SPL is plotted in green. The recorded sound 
    corresponding to trucks is plotted in red. The recorded sound corresponding to anything other than a truck is 
    plotted in light gray.
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking. Cannot be lower than 3.
        
    smoothing_window : float
        Parameter for smoothing the current SPL. Increasing it smoothes the curve more.
        
    smoothing_window_size_ambient : int
        Parameter for smoothing the ambient SPL. Increasing it smoothes the curve more.
        
    ds_factor : int
        Downsample factor for getting the median.
        
    'smoothing' : String
        Type of smoothing. Either median, mean, or gaussian.
    """
    y_vals = []
    joined_df_reset_index = joined_df_median.reset_index()

    #Peaks for all SPL values
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df_median['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    dBAS_peaks = joined_df_median['dBAS'][spl_peaks]
    timestamp_peaks = joined_df_reset_index['index'][spl_peaks]
   
    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    truck_dBAS_peaks = spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1] 
    
    #Peaks for SPL values corresponding to other clusters
    other_dBAS_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['dBAS']
    other_timestamp_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['index']
    
    fig, ax1 = plt.subplots()  
    color = 'b'
    
    ax1.set_xlabel('time')
    ax1.set_ylabel('SPL', color=color)
        
    #Plot dot over max SPL value when that value corresponds to a trucking instance
    ax1.scatter(truck_timestamp_peaks, truck_dBAS_peaks, color='r', s=20)
    ax1.tick_params(axis='y', labelcolor=color)
    
    #Plot SPL peaks corresponding to other clusters
    ax1.scatter(other_timestamp_peaks, other_dBAS_peaks, color='tab:gray', s=20)
    ax1.set_xlim([joined_df_reset_index['index'].iloc[0], joined_df_reset_index['index'].iloc[-1]])
    
    if smoothing == 'median':
        plot_y = scipy.signal.medfilt(joined_df_median['dBAS'].values, smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.signal.medfilt(joined_df_median['median_dBAS'].values, \
                                             smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'mean':
        plot_y = scipy.ndimage.convolve(joined_df_median['dBAS'].values, 
                                             np.ones(smoothing_window_size) / smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.convolve(joined_df_median['median_dBAS'].values, \
                                                    np.ones(smoothing_window_size_ambient) / smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'gaussian':
        plot_y = scipy.ndimage.filters.gaussian_filter(joined_df_median['dBAS'].values, \
                                                                smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.filters.gaussian_filter(joined_df_median['median_dBAS'].values, \
                                                                smoothing_window_size_ambient)[::ds_factor]
    else:
        raise Exception('Unknown smoothing type')
        
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], plot_y, color='g')
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], plot_y_median, color='b')
    
    #Plot regions corresponding to trucks
    ax1.fill_between( \
                     x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=plot_y_median, \
                     y2=plot_y, \
                     where=(joined_df_median.iloc[::ds_factor]['assignment']==1) & (plot_y>plot_y_median), \
                     color='r', alpha=0.5, interpolate=True)
    
    #Plot regions corresponding to other clusters
    ax1.fill_between(x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=plot_y_median, \
                     y2=plot_y, \
                     where=(joined_df_median.iloc[::ds_factor]['assignment']==2) & (plot_y>plot_y_median), 
                     color='tab:gray', alpha=0.5, interpolate=True)
    
    ax1.set_ylim(joined_df_median['dBAS'].min())
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()

def plot_truck_clusters_final(joined_df_median, peak_window_size, \
                                       smoothing_window_size, smoothing_window_size_ambient, ds_factor, smoothing):
    """
    This is the modeling function used in the presentation.
    Plots the SPL over time with visual cues indicating the presence of trucks over time for one sensor. The shaded
    regions indicate time when sound was recorded (versuss just SPL, which is constantly recorded). There is only
    shading when the SPL recorded at the time is greater than the median SPL. 
    There are dots indicating SPL peaks that correspond to truck activity, plotted in red. The median SPL is plotted 
    in dark gray. The SPL is plotted in light gray. The recorded sound corresponding to trucks is plotted in red. The
    recorded sound corresponding to anything other than a truck is plotted in light blue.
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking. Cannot be lower than 3.
        
    smoothing_window : float
        Parameter for smoothing the current SPL. Increasing it smoothes the curve more.
        
    smoothing_window_size_ambient : int
        Parameter for smoothing the ambient SPL. Increasing it smoothes the curve more.
        
    ds_factor : int
        Downsample factor for getting the median.
        
    'smoothing' : String
        Type of smoothing. Either median, mean, or gaussian.
    """
    y_vals = []
    joined_df_reset_index = joined_df_median.reset_index()
    
    #Peaks for all SPL values
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df_median['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    dBAS_peaks = joined_df_median['dBAS'][spl_peaks]
    timestamp_peaks = joined_df_reset_index['index'][spl_peaks]

    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    truck_dBAS_peaks = spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1]
    
    other_dBAS_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['dBAS']
    other_timestamp_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['index']
    
    fig, ax1 = plt.subplots()  
    
    ax1.set_xlabel('Time', fontweight='bold', fontsize=25)
    ax1.set_ylabel('SPL (dB)', fontweight='bold', fontsize=25)
    ax1.set_title('SPL Levels', fontweight='bold', fontsize=30)   
    
    #Plot dot over max SPL value when that value corresponds to a trucking instance
    ax1.scatter(truck_timestamp_peaks, truck_dBAS_peaks, color='r', s=50)
    ax1.tick_params(labelsize=20)
    
#     #Plot SPL peaks corresponding to other clusters
#     ax1.scatter(other_timestamp_peaks, other_dBAS_peaks, color='tab:gray', s=20)
# #     ax1.plot(sliced_joined_df_reset_index['index'], sliced_joined_df['dBAS'])
    ax1.set_xlim([joined_df_reset_index['index'].iloc[0], joined_df_reset_index['index'].iloc[-1]])
    
    if smoothing == 'median':
        plot_y = scipy.signal.medfilt(joined_df_median['dBAS'].values, smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.signal.medfilt(joined_df_median['median_dBAS'].values, \
                                             smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'mean':
        plot_y = scipy.ndimage.convolve(joined_df_median['dBAS'].values, 
                                             np.ones(smoothing_window_size) / smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.convolve(joined_df_median['median_dBAS'].values, \
                                                    np.ones(smoothing_window_size_ambient) / smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'gaussian':
        plot_y = scipy.ndimage.filters.gaussian_filter(joined_df_median['dBAS'].values, \
                                                                smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.filters.gaussian_filter(joined_df_median['median_dBAS'].values, \
                                                                smoothing_window_size_ambient)[::ds_factor]
    else:
        raise Exception('Unknown smoothing type')
    
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                 plot_y, color='lightgray', linewidth=3)
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                 plot_y_median, color='dimgray', linewidth=3)
    
#     Plot regions corresponding to other clusters
    other_where = (joined_df_median.iloc[::ds_factor]['assignment']==2) & (plot_y > plot_y_median)
    other_where |= np.roll(other_where, 1)
    ax1.fill_between(x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=plot_y_median, \
                     y2=plot_y, \
                     where=other_where, 
                     color='lightblue', interpolate=False)

    
    #Plot regions corresponding to trucks
    truck_where = (joined_df_median.iloc[::ds_factor]['assignment']==1) & (plot_y > plot_y_median)
    truck_where |= np.roll(truck_where, 1)
    ax1.fill_between(x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=plot_y_median, \
                     y2=plot_y, \
                     where=truck_where, \
                     color='r', interpolate=False)

    ax1.set_ylim(joined_df_median['median_dBAS'].min())
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()

def plot_truck_clusters_normalized(joined_df_median, peak_window_size, \
                                       smoothing_window_size, smoothing_window_size_ambient, ds_factor, smoothing):
    """
    Plots the SPL normalized to the median SPL over time with visual cues indicating the presence of trucks over time for 
    one sensor. The shaded regions indicate time when sound was recorded (versus just SPL, which is constantly recorded). 
    There is only shading when the SPL recorded at the time is greater than the median SPL. 
    There are dots indicating SPL peaks that correspond to truck activity, plotted. The SPL is plotted in green. The 
    recorded sound corresponding to trucks is plotted in red. The recorded sound corresponding to anything other than 
    a truck is plotted in light blue.
    
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking. Cannot be lower than 3.
        
    smoothing_window : float
        Parameter for smoothing the current SPL. Increasing it smoothes the curve more.
        
    smoothing_window_size_ambient : int
        Parameter for smoothing the ambient SPL. Increasing it smoothes the curve more.
        
    ds_factor : int
        Downsample factor for getting the median.
        
    'smoothing' : String
        Type of smoothing. Either median, mean, or gaussian.
    """
    y_vals = []
    joined_df_reset_index = joined_df_median.reset_index()
    
    joined_df_difference = joined_df_median['dBAS'] - joined_df_median['median_dBAS']
    for i,x in enumerate(joined_df_difference):
        if x < 0:
            joined_df_difference[i] = 0

    #Peaks for all SPL values
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df_median['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    dBAS_peaks = joined_df_median['dBAS'][spl_peaks]
    timestamp_peaks = joined_df_reset_index['index'][spl_peaks]
   
    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    print(truck_timestamp_peaks)
    truck_dBAS_peaks = spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1]
    print(truck_dBAS_peaks)
    
    
    #Peaks for SPL values corresponding to other clusters
    other_dBAS_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['dBAS']
    other_timestamp_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['index']
    
    fig, ax1 = plt.subplots()  
    color = 'b'
    
    ax1.set_xlabel('time')
    ax1.set_ylabel('SPL', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim([joined_df_reset_index['index'].iloc[0], joined_df_reset_index['index'].iloc[-1]])
    
    if smoothing == 'median':
        plot_y = scipy.signal.medfilt(joined_df_median['dBAS'].values, smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.signal.medfilt(joined_df_median['median_dBAS'].values, \
                                             smoothing_window_size_ambient)[::ds_factor]
        normalized_plot_y = plot_y - plot_y_median
    elif smoothing == 'mean':
        plot_y = scipy.ndimage.convolve(joined_df_median['dBAS'].values, 
                                             np.ones(smoothing_window_size) / smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.convolve(joined_df_median['median_dBAS'].values, \
                                                    np.ones(smoothing_window_size_ambient) / \
                                                    smoothing_window_size_ambient)[::ds_factor]
        normalized_plot_y = plot_y - plot_y_median       
    elif smoothing == 'gaussian':
        plot_y = scipy.ndimage.filters.gaussian_filter(joined_df_median['dBAS'].values, \
                                                                smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.filters.gaussian_filter(joined_df_median['median_dBAS'].values, \
                                                                smoothing_window_size_ambient)[::ds_factor]
        normalized_plot_y = plot_y - plot_y_median
    else:
        raise Exception('Unknown smoothing type')
    
    #For areas where SPL is lower than the median, change it to 0
    for i,x in enumerate(normalized_plot_y):
        if x < 0:
            plot_y[i] = 0
            
    #Plot dot over max SPL value when that value corresponds to a trucking instance
    normalized_truck_peaks = joined_df_difference.loc[truck_timestamp_peaks]
    ax1.scatter(truck_timestamp_peaks, normalized_truck_peaks, color='r', s=20)
    
    ax1.plot(joined_df_median.iloc[::ds_factor].reset_index()['index'], normalized_plot_y, color='g')
    
    #Plot regions corresponding to trucks
    ax1.fill_between( \
                     x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(plot_y_median)*[0], \
                     y2=normalized_plot_y, \
                     where=(joined_df_median.iloc[::ds_factor]['assignment']==1), \
                     color='r', alpha=0.5, interpolate=True)
    
    #Plot regions corresponding to other clusters
    ax1.fill_between(x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(plot_y_median)*[0], \
                     y2=normalized_plot_y, \
                     where=(joined_df_median.iloc[::ds_factor]['assignment']==2), 
                     color='tab:gray', alpha=0.5, interpolate=True)
    
    ax1.set_ylim(0)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()

def plot_truck_clusters_normalized_final(joined_df_median, peak_window_size, \
                                       smoothing_window_size, smoothing_window_size_ambient, ds_factor, smoothing):
    """
    Plots the SPL normalized to the median.
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking. Cannot be lower than 3.
        
    smoothing_window : float
        Parameter for smoothing the current SPL. Increasing it smoothes the curve more.
        
    smoothing_window_size_ambient : int
        Parameter for smoothing the ambient SPL. Increasing it smoothes the curve more.
        
    ds_factor : int
        Downsample factor for getting the median.
        
    'smoothing' : String
        Type of smoothing. Either median, mean, or gaussian.
    """
    y_vals = []
    joined_df_reset_index = joined_df_median.reset_index()

    #Peaks for all SPL values
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df_median['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    dBAS_peaks = joined_df_median['dBAS'][spl_peaks]
    timestamp_peaks = joined_df_reset_index['index'][spl_peaks]
   
    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    truck_dBAS_peaks = spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1]
    print(len(truck_timestamp_peaks))
    
    #Peaks for SPL values corresponding to other clusters
    other_dBAS_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['dBAS']
    other_timestamp_peaks = spl_peaks_arr.loc[spl_peaks_arr['assignment']==2]['index']
    
    fig, ax1 = plt.subplots()  
    
    ax1.set_xlabel('Time (One Month)', fontweight='bold', fontsize=25)
    ax1.set_ylabel('Increase Above Ambient SPL (dB)', fontweight='bold', fontsize=25)
    ax1.set_title('Truck Noise Contribution Over June', fontweight='bold', fontsize=30)
        
    #Plot all SPL peaks
    ax1.tick_params(labelsize='20')
    
    #Plot SPL peaks corresponding to other clusters
    ax1.set_xlim([joined_df_reset_index['index'].iloc[0], joined_df_reset_index['index'].iloc[-1]])
    
    if smoothing == 'median':
        plot_y = scipy.signal.medfilt(joined_df_median['dBAS'].values, smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.signal.medfilt(joined_df_median['median_dBAS'].values, \
                                               smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'mean':
        plot_y = scipy.ndimage.convolve(joined_df_median['dBAS'].values, 
                                             np.ones(smoothing_window_size) / smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.convolve(joined_df_median['median_dBAS'].values, \
                                                    np.ones(smoothing_window_size_ambient) / \
                                                    smoothing_window_size_ambient)[::ds_factor]
    elif smoothing == 'gaussian':
        plot_y = scipy.ndimage.filters.gaussian_filter(joined_df_median['dBAS'].values, \
                                                                smoothing_window_size)[::ds_factor]
        plot_y_median = scipy.ndimage.filters.gaussian_filter(joined_df_median['median_dBAS'].values, \
                                                                smoothing_window_size_ambient)[::ds_factor]
    else:
        raise Exception('Unknown smoothing type')
        
    normalized_plot_y = plot_y - plot_y_median
        
    for i,x in enumerate(normalized_plot_y):
        if x < 0:
            normalized_plot_y[i] = 0
    
#     Plot regions corresponding to trucks
    truck_where = (joined_df_median.iloc[::ds_factor]['assignment']==1)
    truck_where |= np.roll(truck_where, 1)
    ax1.fill_between( \
                     x=joined_df_median.iloc[::ds_factor].reset_index()['index'], \
                     y1=len(plot_y_median)*[0], \
                     y2=normalized_plot_y, \
                     where=truck_where, \
                     color='r', alpha=0.5, interpolate=True)
    
#   Plots vertical lines to mark each week
    ax1.vlines(joined_df_median.reset_index()['index'][::604800], \
               0, plot_y.max() + 5, color='dimgray')
    ax1.set_ylim(0, normalized_plot_y.max())
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout()
    fig.show()
