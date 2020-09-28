# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:09:20 2020

@author: olehe

This script first does KMeans clustering, and the descriptive statistics.



"""


#%% Imports


import os
import datetime
import time
import dask.dataframe as dd
from dask.distributed import Client
from dask.multiprocessing import get

import pandas as pd
import numpy as np
from sklearn import preprocessing

from dask.array import stats
import dask.array
import dask_ml.cluster

import seaborn as sns
from matplotlib import pyplot as plt

# Start a dask-client.
client = Client(n_workers=1, threads_per_worker=7, processes=False, memory_limit='40GB', dashboard_address=9002)

#%% Preprocessing

# Uncomment if it's needed to process the data again
'''
spotifyData = dd.read_csv('*AndUnique_allFeatures.csv', assume_missing=True)

# Mean and std per unique hour of the week.
hourAvgData = spotifyData.groupby('unique').agg({
	'beat_strength':['mean', 'std'],
	'loudness':['mean', 'std'],
	'mechanism':['mean', 'std'],
	'organism':['mean', 'std'],
	'tempo':['mean', 'std'],
	'acousticness':['mean', 'std'],
	'bounciness':['mean', 'std'],
	'danceability':['mean', 'std'],
	'dyn_range_mean':['mean', 'std'],
	'energy':['mean', 'std'],
	'flatness':['mean', 'std'],
	'instrumentalness':['mean', 'std'],
	'liveness':['mean', 'std'],
	'speechiness':['mean', 'std'],
	'valence':['mean', 'std'],
	}).compute()

'''

#%% Kmeans-clustering


# Read already-processed data.
hourAvgData = pd.read_pickle('hourAvgDataPickle')
hourAvgValues = hourAvgData.values


# Cluster only on the means
hourAvgValues = hourAvgValues[:,[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]]


# Scale data
scaler = preprocessing.StandardScaler()
scaler.fit(hourAvgValues)
hourAvgScaled = scaler.transform(hourAvgValues)




# K-means loop for selecting optimal k.
SumMetric = []
K = range(2, 25)
for k in K:
	km = dask_ml.cluster.KMeans(n_clusters=k, init_max_iter=k, oversampling_factor=2, max_iter=100)
	km.fit(hourAvgScaled)
	thisMetric = km.inertia_
	SumMetric.append(thisMetric)


# Inspect the k-means
metricPlot = sns.lineplot(K, SumMetric)
metricPlot.set_xticks(K)
# 5 is optimal K.

# Re-run k-means with a higher max_iter
skm_5 = dask_ml.cluster.KMeans(n_clusters=5, init_max_iter=5, oversampling_factor=2, max_iter=1000)
skm_5.fit(hourAvgScaled)
skm_5_labels = skm_5.labels_.compute()




#%% Save results

cc5 = pd.DataFrame(scaler.inverse_transform(skm_5.cluster_centers_), columns=['bsMean',
	'loudMean',
	'mechMean',
	'orgMean',
	'tempMean',
	'acousMean',
	'bouncMean', 
	'danceMean',
	'dynMean',
	'enerMean',
	'flatMean',
	'instMean',
	'liveMean',
	'speechMean',
	'valMean'])
cc5.to_pickle('SML_clusterCentroids_k5')



#%% Calculate how these relates to the mean.
	
hourMeanAll = hourAvgValues.mean(axis=0)

ccRelToMean = cc5 - hourMeanAll

	
#%% Calculate descriptive statistics per subdivision

labels=pd.read_pickle('kmeansLabelsForKequal5')

spotifyData = dd.read_csv('*AndUnique_allFeatures.csv', assume_missing=True)

# first, need to groupby the actual data-subdivisions
# Note that these subdivision have different labels than what is used in the manuscript
subMorning = labels.index[labels[0]==4].tolist()
subMidnight = labels.index[labels[0]==2].tolist()
subAfternoon = labels.index[labels[0]==1].tolist()
subEvening = labels.index[labels[0]==3].tolist()
subEarlyMorning = labels.index[labels[0]==0].tolist()


# Need to do this per subdivision for disk space reasons.

subMorningData = spotifyData[spotifyData['unique'].isin(subMorning)][['danceability',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()

subMorningData.to_csv('MorningDataDistribution.csv')



subMidnightData = spotifyData[spotifyData['unique'].isin(subMidnight)][['danceability',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()
subMidnightData.to_csv('MidnightDataDistribution.csv')

subAfternoonData = spotifyData[spotifyData['unique'].isin(subAfternoon)][['danceability',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()
subAfternoonData.to_csv('AfternoonDataDistribution.csv')


#still need to do evening
subEveningData = spotifyData[spotifyData['unique'].isin(subEvening)][['danceability',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()
subEveningData.to_csv('EveningDataDistribution.csv')



#early morning seems ok.
subEarlyMorningData = spotifyData[spotifyData['unique'].isin(subEarlyMorning)][['danceability',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()

subEarlyMorningData.to_csv('EarlyMorningDataDistribution.csv')








