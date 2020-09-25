# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:09:20 2020

@author: olehe



This should do first some summary statistics, and/or at the same time some
clustering of the Spotify data. Use that as a "starting point" for the eventual
machine learning.

Also, get new dataset. Beware that not everything is available on the API anymore.


These are the available features in api:
	acousticness
	danceability
	energy
	instrumentalness
	liveness
	loudness
	speechiness
	valence
	tempo
	
These are the ones available in the dataset:
	beat_strength
	loudness
	mechanism
	organism
	tempo
	acousticness
	bounciness
	danceability
	dyn_range_mean
	energy
	flatness
	instrumentalness
	liveness
	speechiness
	valence


"""


#%%
# Import everything


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


client = Client(n_workers=1, threads_per_worker=7, processes=False, memory_limit='40GB', dashboard_address=9002)

# Uncomment if it's needed to process the data again
#spotifyData = dd.read_csv('*AndUnique_allFeatures.csv', assume_missing=True)



#%% Summary data for reduced KMEANS

'''
# This section does the averaging per unique hour of the week
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

# Read already-processed data.

hourAvgData = pd.read_pickle('hourAvgDataPickle')
hourAvgValues = hourAvgData.values


# Select only the means, and not the std's

#hourAvgScaled = hourAvgScaled[:,[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]]
hourAvgValues = hourAvgValues[:,[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]]


# Now scale the data.
# should use standardscaler so that we can go back to original data from
# the cluster centroids

scaler = preprocessing.StandardScaler()
scaler.fit(hourAvgValues)

hourAvgScaled = scaler.transform(hourAvgValues)




# just a quick k-means loop for optimal k

SumMetric = []
K = range(2, 25)

for k in K:
	km = dask_ml.cluster.KMeans(n_clusters=k, init_max_iter=k, oversampling_factor=2, max_iter=100)
	km.fit(hourAvgScaled)
	thisMetric = km.inertia_
	SumMetric.append(thisMetric)


metricPlot = sns.lineplot(K, SumMetric)
metricPlot.set_xticks(K)
# 5 is optimal K.

# now k-means

skm_5 = dask_ml.cluster.KMeans(n_clusters=5, init_max_iter=5, oversampling_factor=2, max_iter=1000)


skm_5.fit(hourAvgScaled)

skm_5_labels = skm_5.labels_.compute()




#%% now save the cluster centroids
# save in transformed mode


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



#%%
# just plot together for fun. Need to melt it though.
scaledDataFrame = pd.DataFrame(hourAvgScaled, index=range(0,168), columns=['bsMean',
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
scaledDataFrame['hour'] = range(0,168)

plt.figure(figsize=(22,8))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['bsMean', 'loudMean', 
													  'mechMean','orgMean',
													  'tempMean', 'acousMean',
													  'bouncMean', 'danceMean',
													  'dynMean', 'enerMean',
													  'flatMean', 'instMean',
													  'liveMean', 'speechMean',
													  'valMean','hour']], ['hour']),
													   label=None)
allPlot.legend_.remove()
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('K-means clustering of MIR-data, k=5')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
	
#%%
# Calculate how these relates to the mean.
	
hourMeanAll = hourAvgValues.mean(axis=0)

ccRelToMean = cc5 - hourMeanAll

	
#%%
# Now calculating the percentiles for the actual data,
# This is needed for getting the right values out at a later point.
# Calculate: min, 5, 20, 80, 95, max.
# can use describe(percentiles=[0, 0.05, 0.2, 0.5, 0.8, 0.95, 1])
# First, just saving the labelling

#labels=pd.DataFrame(skm_5_labels)	
#labels.to_pickle('kmeansLabelsForKequal5')
labels=pd.read_pickle('kmeansLabelsForKequal5')

spotifyData = dd.read_csv('*AndUnique_allFeatures.csv', assume_missing=True)

# first, need to groupby the actual data-subdivisions
# beware that these numbers, used in the k-means may occasionally change.
subMorning = labels.index[labels[0]==4].tolist()
subMidnight = labels.index[labels[0]==2].tolist()
subAfternoon = labels.index[labels[0]==1].tolist()
subEvening = labels.index[labels[0]==3].tolist()
subEarlyMorning = labels.index[labels[0]==0].tolist()


# This seems to be the only reasonable way of doing it.

# Careful, runs out of space on disk...
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
# started 12:17
# finished maye 1630ish


#started this at 16:47.
# this fails, need to do them one by one...


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








