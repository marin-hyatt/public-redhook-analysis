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
