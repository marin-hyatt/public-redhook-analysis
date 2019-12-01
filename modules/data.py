import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import datetime 
from datetime import timedelta
import pytz
from pytz import timezone
from numpy import load
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
import h5py
import pylab
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import librosa
import matplotlib.dates as md
import sys

def create_hdf5(sensor_timestamps, sensor_id, sensor_timestamps_orig, file_path, sensor_data):
    """
    Creates an hdf5 file with all of the data needed for analysis and visualization.
    
    Parameters
    ----------
    sensor_timestamps : numpy array
        An array of timestamps (not converted into datetime format) for each sound sample. This timestamp is adjusted to reflect
        the actual time in which the sample was recorded (not the time in which the 10 second recording was made).
    
    sensor_id : numpy array
        An array containing the sensor ID for each sample.
        
    sensor_timestamps_orig : numpy array
        An array of timestamps (not converted into datetime format) for each sound sample. This timestamp is not adjusted, 
        meaning that it has the time that the 10 second recording was taken instead of the time of the acutal 1-second sample.
        
    file_path : numpy array
        An array of the file paths of each sample. Should look like
        "Users/marin/redhook/embeddings/sonycnodeb827ebc178d2.sonyc/2019-06-01/08/1559390446.46.npz".
    
    sensor_data : numpy array
        An array of the 512-dimension embedding for each sample.
        
    """
    sensor_timestamps = np.load('sensor_timestamps_arr.npy')
    sensor_id = np.load('sensor_id_arr.npy')
    sensor_timestamps_orig = np.load('sensor_timestamps_orig_arr.npy')
    file_path = np.load('file_path_arr.npy')

    sensor_data = np.load('sensor_data_arr.npy')

    #Creating the hdf5 file
    with h5py.File('sound_data_improved.hdf5', 'w') as h5:
        d = h5.create_dataset('sound_data',
                              (3388858,),
                              dtype=[('timestamp', 'f8'),
                                     ('timestamp_orig', 'f8'),
                                     ('sensor_id', 'S128'),
                                     ('feature_vector', 'f4', (512,)),
                                     ('file_path', 'S128')
                                    ],
                                chunks=True,
                                maxshape=(3388858 * 516,))
        for idx in range(sensor_data.shape[0]):
            d[idx] = (sensor_timestamps[idx], sensor_timestamps_orig[idx], sensor_id[idx], sensor_data[idx], file_path[idx])


def create_dataframe(output_path, hdf5_path, csv_path, start_date, end_date, sensor_name, sample_size=1000, num_dimensions=45, \
                     num_clusters=64, truck_clusters=[5, 10, 11, 18, 20, 37, 42, 57, 63]):
    """
    Creates a dataframe with a datetime index between the specified start and end dates. It has columns for SPL, median SPL,
    and whether or not the noise is a truck at that time (indicated numerically).
    
    Parameters
    ----------
    output_path : String
        A filepath to store the dataframe in. Example:../output/june_2019_df.csv.
        
    hdf5_path : String
        A filepath to an hdf5 file. This file should have a dataset named 'sound_data' and should have columns named
        'timestamp', 'timestamp_orig', 'sensor_id', 'feature_vector', and 'file_path'.
        
    csv_path : String
        A filepath to a csv file. This file should have a column for timestamps (in UTC format) and a column for SPL 
        corresponding to that timestamp.
        
    start_date : datetime
        An example of this parameter is datetime.datetime(2019, 6, 1, 4). Used to determine the start date of the dataframe,
        but only if it is after the start date of the SPL csv file (because SPL should always be present in the visualizations).
    
    end_date : datetime
        An example of this parameter is datetime.datetime(2019, 6, 30, 4). Used to determine the end date of the dataframe.
        
    sensor_name : String
        The name of the sensor to get sound data for. An example is b'sonycnode-b827ebc178d2.sonyc'.
        
    sample_size : int
        The size of the subsample of the data used for computational efficiency in clustering. Default is 10,000.
        
    num_dimensions: int
        The number of dimensions to reduce the feature vector to using PCA. Default is 45.
        
    num_clusters: int
        The number of clusters to group the data into. Default is 64, and it shouldn't really be changed unless someone has
        listened to sound samples from each cluster and identified which clusters correspond to trucks.
        
    truck_clusters: array of ints
    	The clusters that correspond to truck sounds. These need to be determined by listening to sound samples from each
        cluster manually.
    """
    
    h5 = h5py.File(hdf5_path, 'r')

    d = h5['sound_data']
    
    print('done with reading hdf5 file')

    # Creating subsample of 10000 points from all four sensors

    sample_nums = np.random.choice(range(d.shape[0]), sample_size, replace = False)

    index = np.zeros(d.shape[0]).astype('bool')
    index[sample_nums] = True
    
    print('done with indexing')
    
    # Reading SPL data from csv

    df = pd.read_csv(csv_path, skiprows = 2, low_memory = False)
    print(df.head())

    time_arr = np.empty(df.shape[0], dtype = datetime.datetime)
    timestamp_arr = df['timestamp'].values
    dBAS_arr = df['dBAS'].values
    
    time_arr = [convert_timestamps(x) for x in timestamp_arr]
    
    time_df = df
    time_df['timestamp'] = time_arr
    
    print('done with creating spl dataframe')
    print(time_df.head())
    
    # Reducing dimensionality

    pca_45 = sklearnPCA(num_dimensions)
    projected_45 = pca_45.fit_transform(d['feature_vector'])
    
    print('done with PCA')

    sensor_mask = get_sensor_mask(d['sensor_id'], sensor_name)
    sensor_transformed = projected_45[sensor_mask]
    sensor_timestamps = d[sensor_mask, 'timestamp']
    sensor_timestamps_dt = [convert_timestamps(x) for x in sensor_timestamps]

    time_arr = [convert_timestamps(x) for x in timestamp_arr]
    
    #Making dataframe with timestamps and cluster assignment
    
    #Getting cluster assignments
    all_cluster_assignments = get_cluster_assignments(num_clusters, sensor_transformed, projected_45[index])
    seconds_sensor_timestamps_dt = [x.replace(microsecond=0) for x in sensor_timestamps_dt]
    
    print('done with getting cluster assignments')
	
    #Creating the dataframe
    assignments_df = pd.DataFrame(data={'assignment':all_cluster_assignments}, \
                                  index = seconds_sensor_timestamps_dt)
    assignments_df.head()
    
    print('done with creating assignments_df')
    print(assignments_df.head())
    
    #Removing duplicate entries
    removed_assignments_df = assignments_df[~assignments_df.index.duplicated()]
	
    #Making a complete date range from start date to end date
    #complete = pd.date_range(start_date, end_date, periods=3600*24*30)
    #seconds_complete = [x.replace(microsecond=0, nanosecond=0) for x in complete]
    #aware_seconds_complete = [pytz.utc.localize(x) for x in seconds_complete]
    #tz_seconds_complete = [x.astimezone(pytz.timezone('US/Eastern')) for x in aware_seconds_complete]

    #Creating a dataframe with SPL values indexed by time
    naive_time_df = [x.replace(tzinfo=None) for x in time_df['timestamp']]
    seconds_complete_timestamp = [x.replace(microsecond=0) for x in time_df['timestamp']]
    spl_df = pd.DataFrame(data={'dBAS': dBAS_arr}, index=seconds_complete_timestamp)
    
    print('done with creating SPL dataframe with time index')
    
    #Joining the SPL dataframe and the dataframe with assignments. This new dataframe uses the index of spl_df.
    all_joined_df = spl_df.join(removed_assignments_df, how='left')
    
    #Replacing all instances of a truck cluster with 1, every other cluster assignment with 2, and the spots with no sound data 
    #with 0.
    all_joined_df = all_joined_df.replace(truck_clusters, 1)
    all_joined_df = all_joined_df.replace(range(2,64), 2)

    all_joined_df.loc[pd.isnull(all_joined_df['assignment']), 'assignment'] = 0
    
    print('done with joining dataframe and replacing cluster assignments')
	
    #Creating matrix of SPL values in each day (to get median SPL)
    spl_complete = spl_df['dBAS']

#     beginning_spl_indices = \
#     pd.date_range(datetime.datetime(2019, 6, 1, 4, 0, 0), datetime.datetime(2019, 6, 1, 4, 0, 42), periods=42)
#     beginning_spl_indices = [x.replace(microsecond=0, nanosecond=0) for x in beginning_spl_indices]
#     beginning_spl_indices = [pytz.utc.localize(x) for x in beginning_spl_indices]
#     beginning_spl_indices = [x.astimezone(pytz.timezone('US/Eastern')) for x in beginning_spl_indices]

#     beginning_spl = pd.Series(np.nan, index=beginning_spl_indices)

#     spl_complete_2 = pd.concat([beginning_spl, spl_complete])
    
#     print('done with creating complete SPL dataframe')
    
#     #Series of SPL for the whole month, without the 42 second shift forward
#     spl_complete_month = spl_complete_2[:-43]
    
    #Creating median array of weekends
    #Creating arrays of SPL collected on weekdays and weekends

    spl_weekends = spl_complete[spl_complete.index.dayofweek >= 5]
    spl_weekdays = spl_complete[spl_complete.index.dayofweek < 5]

    #Removing duplicate times from both arrays

    spl_weekends = spl_weekends[~spl_weekends.index.duplicated()]
    spl_weekdays = spl_weekdays[~spl_weekdays.index.duplicated()]

    #Getting median values for weekends and weekdays

    weekend_medians = get_median(spl_weekends)
    print('done with getting weekend medians')
    weekday_medians = get_median(spl_weekdays)
    print('done with getting weekday medians')
    
    #Creating an array of the same size as the arrays in all_joined_df, and repeatedly filling it with median values

    #Creating datetime indices spanning the whole month

    medians_df_index = all_joined_df.reset_index()['index']

    #Filling an array that spans the whole month, then filling it with median values for one day that are repeated
    month_weekday_median = np.empty(len(medians_df_index))
    for i in range(len(medians_df_index)):
        month_weekday_median[i] = weekday_medians[i%len(weekday_medians)]
        
    print('done with creating and filling median array')

    #Creating dataframe with weekday medians for whole month

    weekday_medians_df = pd.DataFrame({'median_dBAS':month_weekday_median}, index=medians_df_index)
    weekend_medians_df_indices = weekday_medians_df.loc[weekday_medians_df.index.dayofweek>=5].index
    weekend_medians_df_values = np.empty(len(weekend_medians_df_indices))

    #Replacing weekday values in weekend times with the weekend median SPL

    for x in range(len(weekend_medians_df_values)):
        weekend_medians_df_values[x] = weekend_medians[x % len(weekend_medians)]

    #Making a dataframe with weekend medians

    weekend_medians_df = pd.DataFrame({'median_dBAS':weekend_medians_df_values}, index=weekend_medians_df_indices)
    weekend_medians_df = weekend_medians_df[~weekend_medians_df.index.duplicated()]
    
    print('done with creating dataframe for weekday and weekend medians')
    
    # Replacing weekend values in dataframe with correct values

    #Merging dataframe with weekday median values and dataframe with weekend median values. The dataframe with weekday median
    #values has indices for the whole month. Indices that are on the weekend will be replaced with the weekend median values.

    both_medians_df = weekday_medians_df.merge(weekend_medians_df, how='outer', left_index=True, right_index=True)

    #median_dBAS_x is the weekday median values, median_dBAS_y is the weekend median values. Since this is an outer join, many
    #values in median_dBAS_y will be NaN.

    #Creates a new column that replaces weekday median values that are in the weekend index with the weekend median values.

    both_medians_df['median_dBAS'] = \
    both_medians_df['median_dBAS_x'].where(both_medians_df['median_dBAS_y'].isnull(), \
                                                                                  both_medians_df['median_dBAS_y'])
    #Gets rid of the columns with weekday and weekend median values, since we have a column with the correct median values for
    #both weekdays and weekends.

    both_medians_df = both_medians_df.drop(['median_dBAS_x', 'median_dBAS_y'], axis=1)
    
    print('done with joining dataframes and merging weekday and weekend data')
    
    # Joining weekday and weekend medians to dataframe

    #Joining the median dataframe to the dataframe with SPL and cluster assignment

    all_joined_df_median = all_joined_df.join(both_medians_df)

    all_joined_df_median.tail()

    removed_all_joined_df_median = all_joined_df_median[~all_joined_df_median.index.duplicated()]
    
    print('done with creating final dataframe')

    #Saving dataframe to csv for later use

    removed_all_joined_df_median.to_csv(output_path)
    
    print('done with saving dataframe to output')

def get_truck_peaks(joined_df_median, peak_window_size):
    """
    Returns a dataframe with the timestamp and SPL value for peaks in SPL that correspond to trucks.
    
    Parameters
    ----------
    joined_df : dataframe
        A dataframe containing timestamps, a column for cluster assignments, and dBAS values.
    
    peak_window_size : int
        Parameter for peak picking. Cannot be lower than 3.
        
    Returns
    -------
    truck_peaks_df : Dataframe
        A dataframe indexed by time
    """
    joined_df_reset_index = joined_df_median.reset_index()
    window = int((peak_window_size-1)/2)
    spl_peaks = librosa.util.peak_pick(joined_df_median['dBAS'], window, window, window, window, 3, 0)
    spl_peaks_arr = joined_df_reset_index.loc[spl_peaks]
    truck_timestamp_peaks = spl_peaks_arr['index'].loc[spl_peaks_arr['assignment']==1]
    truck_dBAS_peaks = (spl_peaks_arr['dBAS'].loc[spl_peaks_arr['assignment']==1]).to_numpy()
    
    truck_peaks_df = pd.DataFrame(data={'truck_peaks': truck_dBAS_peaks}, index=truck_timestamp_peaks)
    print(truck_dBAS_peaks)
    return truck_peaks_df
    
def get_subsample_mask(num_samples, target_arr):
    """
    Returns a mask to apply on an array of data. The mask represents a random subsample of the data that contains num_samples
    data points.
    
    Parameters
    ----------
    num_samples : int
        The number of samples to randomly select from the data.
        
    target_arr : array
        An array to take the subsample from.
        
    Returns
    -------
    index : boolean array
        Mask to apply to target_arr that subsamples the values. The number of True values is num_samples.
    """
    sample_nums = np.random.choice(range(target_arr.shape[0]), num_samples, replace = False)
    index = np.zeros(target_arr.shape[0]).astype('bool')
    index[sample_nums] = True
    
    return index
    
def convert_timestamps(sensor_timestamp):
    """
    Converts a float timestamp to a datetime object.
    
    Parameters
    ----------
    sensor_timestamp : float
        A timestamp in float form.
        
    Returns
    -------
    dt : datetime object
       Datetime object corresponding to the same time as the float timestamp.
    """
    j = sensor_timestamp
    dt = datetime.datetime.utcfromtimestamp(j)
    dt = pytz.UTC.localize(dt)
    dt = dt.astimezone(pytz.timezone('US/Eastern'))
    return dt

def get_sensor_mask(sensor_name, target_arr):
    """
    Returns a mask to apply on an array of data from different sensors. The mask filters out the data from one sensor.
    
    Parameters
    ----------
    sensor_name : String
        The name of the sensor to retrieve the data for, should take the form of b'sonycnode-[name].sonyc'
    
    target_arr : array of Strings
        Array containing the names of all the sensors.
        
    Returns
    -------
    sensor_mask : boolean array
        A boolean array of the same shape as target_arr, with True values corresponding to the indices containing the
        sensor_name.
    """
    sensor_mask = (target_arr == sensor_name)
    return sensor_mask

def get_time_mask(beginning, end, time_arr):
    """
    Returns a boolean mask to apply to a datetime array, with the goal of returning times between a beginning and 
    end time, including the beginning time but excluding the end time. 
    
    Parameters
    ----------
    beginning : datetime.datetime
        The beginning time to use for the mask.
        
    end : datetime.datetime
        The end time to use for the mask.
        
    time_arr : np array
        A numpy array of naive datetime objects. The mask will be applied to this array.
        
    Returns
    -------
    interval_mask : boolean array
        A boolean array to use as a mask on time_arr.
    """
    interval_mask = (time_arr >= beginning) & (time_arr < end)
    return interval_mask

def get_cluster_assignments(num_clusters, sensor_transformed, fit_arr):
    """
    Returns an array containing the number of each cluster each data point in sensor_transformed is assigned to.
    Clustering is performed using MiniBatchKMeans. Also in display.py because it is needed to make the graphs.
    
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

def convert_to_hour(sensor_timestamp):
    """
    Returns the number of hours since the beginning of the month of a timestamp in Unix format. For example, 
    if the timestamp converts to 9:00 am on June 2nd, 57 (e.g. 48 + 9) would be returned.
    
    Parameters
    ----------
    sensor_timestamp : float
        A Unix timestamp.
        
    Returns
    -------
   total_hour : int
        The number of hours since the beginning of the month of the recording date. 
    """
    j = sensor_timestamp[i]
    dt = datetime.datetime.utcfromtimestamp(j)
    dt = pytz.UTC.localize(dt)
    day = dt.astimezone(pytz.timezone('US/Eastern')).day
    hour = dt.astimezone(pytz.timezone('US/Eastern')).hour
    total_hour = 24*(day-1) + hour
    return total_hour

def minute_average(time_arr, dBAS_arr):
    """
    Averages SPL over each minute and returns the averaged values in an array, along with an array of corresponding timestamps.
    
    Parameters
    ----------
    time_arr : array of timestamps
        An array of timestamps. There should be a timestamp for every second.
    
    dBAS_arr : float array
        Array of SPL values for each timestamp in time_arr. 
        
    Returns
    -------
    minute_time_arr : array of timestamps.
        An array of timestamps. There should be one timestamp for every minute.
        
    minute_dBAS_arr : float array
        An array of SPL values averaged over each minute.
    """
    minute_time_arr = np.empty(int(len(time_arr)/60), dtype = datetime.datetime)
    minute_dBAS_arr = np.empty(int(len(time_arr)/60))

    step = 60
    i = 0
    count = 0
    while i < len(time_arr): 
        minute_time_arr[count] = time_arr[i]
        minute_dBAS_arr[count] = np.average(dBAS_arr[i:i+step])
        i += step
        count += 1
        
    return(minute_time_arr, minute_dBAS_arr)

def get_median(spl_arr):
    """
    Returns an array of median SPL values for each second over a specified set of days in June (weekdays or weekends). For 
    each second of the day, it takes the median SPL over all weekdays, so that there is a medial SPL value for each second. 
    For example, it takes the median SPL value for 4:00:00 a.m. for all weekdays in June.
    
    Parameters
    ----------
    spl_arr : float array with datetime index
        Array with a datetime index and SPL values corresponding to the index.
    """
    
    #Creates an array of all days in the spl arr
    all_day_arr = []
    #print(np.unique(spl_arr.index.day))
    for day in np.unique(spl_arr.index.day):
        day_arr = [x for x in spl_arr.reset_index()['index'] \
                   if x.day==day]
        all_day_arr.append(day_arr)
        #print(str(day) + ': ' + str(len(day_arr)))

    #Creates a matrix of each second of the day and all the days in the all_day_arr, in order to get the median across
    #the days
    day_time_matrix = np.ndarray((86400,len(all_day_arr)))
    for i,day in enumerate(all_day_arr):
        complete_day_arr = np.zeros(86400)
        for time in day:
            num_secs_since_beginning = 3600*time.hour + 60*time.minute + time.second
            complete_day_arr[num_secs_since_beginning] = \
            spl_arr.loc[time]
        #changes zeros to NaN values for the purposes of getting a median
        complete_day_arr[complete_day_arr<1] = np.nan

        day_time_matrix[:,i] = complete_day_arr
    
    median_arr = np.nanmedian(day_time_matrix, axis=1)
    
    return median_arr

