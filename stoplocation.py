from datetime import timedelta
import os
from datetime import datetime
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint,Point
import math
def get_centermost_point(cluster):
    """
    This function returns the center-most point from a cluster by taking a set of points (i.e., a cluster)
    and returning the point within it that is nearest to some reference point (in this case, the cluster's centroid):
    """
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).meters)
    return tuple(centermost_point

def get_neighbors(df,index_center, spatial_eps, temporal_eps):
    """
    :param df: date in field date_event
                gps as float(lat) float(lon)
    :param index_center:
    :param spatial_eps: metres   #TODO: Determine the parameter 'eps' as the median of the distribution of the maximum of the minPts-nearest neighbors distances for each point, knn with k=minPts
    :param temporal_eps: minutes
    :return: point's neighberhood
    """
    neighborhood = []

    center_point = df.loc[index_center]

    # filter by time
    min_time = center_point['date_event'] - timedelta(minutes = temporal_eps)
    max_time = center_point['date_event'] + timedelta(minutes = temporal_eps)
    df = df[(df['date_event'] >= min_time) & (df['date_event'] <= max_time)]


    # filter by distance
    for index, point in df.iterrows():
        if index != index_center:
            distance = great_circle((center_point['lat'], center_point['lon']), (point['lat'], point['lon'])).meters
            if distance <= spatial_eps:
                neighborhood.append(index)

    return neighborhood
def STDBSCAN(df, spatial_eps, temporal_eps, min_neighbors,log):
    """

    :param df:
    :param spatial_eps: metres the median of the distribution of the maximum of the minPts-nearest neighbors distances for each point
    :param temporal_eps: minutes
    :param min_neighbors: log (df.shape)
    :return: clustered df
    """

    cluster_label = 0
    NOISE = -1
    INVISITED = 5555555555
    list = []

    # initialize each point as invisited
    df['cluster'] = INVISITED
    # for each point in database
    for index, point in df.iterrows():
        if df.loc[index]['cluster'] == INVISITED:
            neighborhood = get_neighbors( df,index, spatial_eps, temporal_eps)

            if len(neighborhood) < min_neighbors:
                df.at[index, 'cluster']= NOISE

            else: # found a core point
                cluster_label +=  1 # in order to start numbering at 1
                df.at[index, 'cluster'] = cluster_label # assign a label to core point

                for neighbor in neighborhood : # assign core's label to its neighborhood
                    df.at[neighbor, 'cluster'] = cluster_label
                    list.append(neighbor) # append neighborhood to list


                while len(list)>0  : # find new neighbors from core point neighborhood
                    current_point_index = list.pop()
                    new_neighborhood = get_neighbors(df, current_point_index, spatial_eps, temporal_eps)
                    #cluster_avg = get_center_time_interval(cluster)TODO function has to be improved
                    if len(new_neighborhood) >= min_neighbors: # current_point is a new core
                        for neig_index in new_neighborhood:
                            neig_cluster = df.loc[neig_index]['cluster']
                            if any([neig_cluster == NOISE,  neig_cluster == INVISITED]) :#&(abs(cluster_avg-neig_cluster)<=alpha) :
                                #  verify cluster average before add new point
                                df.at[neig_index, 'cluster']= cluster_label
                                list.append(neig_index)



    return df
def get_db_clusters(df, epsilon, min_samples):
    coords = df_to_coords(df)
    kms_per_radian = 6371.0088
    epsilon = epsilon / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters-1)])
    centermost_points =clusters.map(get_centermost_point)
    lats, lons =zip(*centermost_points)
    df_rep_points = pd.DataFrame({'lon': lons, 'lat': lats})
    return df_rep_points

def get_address(df):
    geolocator = Nominatim(user_agent="zoi_detector_v2")
    df['address'] = df.apply(lambda row: geolocator.reverse((row['lat'], row['lon'])), axis=1)
    return df