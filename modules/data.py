import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import datetime 
from datetime import timedelta
import pytz
from pytz import timezone
import matplotlib.dates as md

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
    	The clusters that correspond to truck sounds. These need to be determined by listening to sound samples from each cluster
        manually.
    """
    
    h5 = h5py.File(hdf5_path, 'r')

    d = h5['sound_data']

    # Creating subsample of 10000 points from all four sensors

    sample_nums = np.random.choice(range(d.shape[0]), sample_size, replace = False)

    index = np.zeros(d.shape[0]).astype('bool')
    index[sample_nums] = True
    
    # Reading SPL data from csv

    df = pd.read_csv(csv_path, skiprows = 2, low_memory = False)

    time_arr = np.empty(df.shape[0], dtype = datetime.datetime)
    timestamp_arr = df['timestamp'].values
    dBAS_arr = df['dBAS'].values
    
    time_df = df
    time_df['timestamp'] = time_arr
    
    # Reducing dimensionality

    pca_45 = sklearnPCA(num_dimensions)
    projected_45 = pca_45.fit_transform(d['feature_vector'])

    sensor_mask = data.get_sensor_mask(d['sensor_id'], sensor_name)
    sensor_transformed = projected_45[sensor_mask]
    sensor_timestamps = d[sensor_mask, 'timestamp']
    sensor_timestamps_dt = [data.convert_timestamps(x) for x in sensor_timestamps]

    time_arr = [data.convert_timestamps(x) for x in timestamp_arr]
    
    #Making dataframe with timestamps and cluster assignment
    
    #Getting cluster assignments
    all_cluster_assignments = data.get_cluster_assignments(num_clusters, sensor_transformed, projected_45[index])
    seconds_sensor_timestamps_dt = [x.replace(microsecond=0) for x in sensor_timestamps_dt]
	
    #Creating the dataframe
    assignments_df = pd.DataFrame(data={'assignment':all_cluster_assignments}, \
                                  index = seconds_sensor_timestamps_dt)
    assignments_df.head()
    
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
    
    #Joining the SPL dataframe and the dataframe with assignments. This new dataframe uses the index of spl_df.
    all_joined_df = spl_df.join(removed_assignments_df, how='left')
    
    #Replacing all instances of a truck cluster with 1, every other cluster assignment with 2, and the spots with no sound data 
    #with 0.
    all_joined_df = all_joined_df.replace(truck_clusters, 1)
    all_joined_df = all_joined_df.replace(range(2,64), 2)

    all_joined_df.loc[pd.isnull(all_joined_df['assignment']), 'assignment'] = 0
	
    #Creating matrix of SPL values in each day (to get median SPL)
    spl_complete = spl_df['dBAS']

    beginning_spl_indices = \
    pd.date_range(datetime.datetime(2019, 6, 1, 4, 0, 0), datetime.datetime(2019, 6, 1, 4, 0, 42), periods=42)
    beginning_spl_indices = [x.replace(microsecond=0, nanosecond=0) for x in beginning_spl_indices]
    beginning_spl_indices = [pytz.utc.localize(x) for x in beginning_spl_indices]
    beginning_spl_indices = [x.astimezone(pytz.timezone('US/Eastern')) for x in beginning_spl_indices]

    beginning_spl = pd.Series(np.nan, index=beginning_spl_indices)

    spl_complete_2 = pd.concat([beginning_spl, spl_complete])
    
    #Series of SPL for the whole month, without the 42 second shift forward
    spl_complete_month = spl_complete_2[:-43]
    
    #Creating median array of weekends
    #Creating arrays of SPL collected on weekdays and weekends

    spl_complete_month_weekends = spl_complete_month[spl_complete_month.index.dayofweek >= 5]
    spl_complete_month_weekdays = spl_complete_month[spl_complete_month.index.dayofweek < 5]

    #Removing duplicate times from both arrays

    spl_complete_month_weekends = spl_complete_month_weekends[~spl_complete_month_weekends.index.duplicated()]
    spl_complete_month_weekdays = spl_complete_month_weekdays[~spl_complete_month_weekdays.index.duplicated()]

    #Getting median values for weekends and weekdays

    weekend_medians = get_median(spl_complete_month_weekends)
    weekday_medians = data.get_median(spl_complete_month_weekdays)
    
    #Creating an array of the same size as the arrays in all_joined_df, and repeatedly filling it with median values
    #Note: this array still has the 42 second shift, but I figured that in the long run it won't matter.
    month_weekday_median = np.empty(len(spl_complete))
    for i in range(len(spl_complete)):
        #Filling with weekday median values first, will replace with weekend median values later
        month_weekday_median[i] = weekday_medians[i%len(weekday_medians)]
        
    # Making dataframes with median values for the month

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
    
    # Joining weekday and weekend medians to dataframe

    #Joining the median dataframe to the dataframe with SPL and cluster assignment

    all_joined_df_cut_median = all_joined_df_cut.join(both_medians_df)

    all_joined_df_cut_median.tail()

    removed_all_joined_df_cut_median = all_joined_df_cut_median[~all_joined_df_cut_median.index.duplicated()]

    #Saving dataframe to csv for later use

    removed_all_joined_df_cut_median.to_csv(output_path)
  #def get_truck_peaks():
    
    
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

