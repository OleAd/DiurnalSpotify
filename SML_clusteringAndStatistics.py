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
# Imports


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


client = Client(n_workers=1, threads_per_worker=4, processes=False, memory_limit='30GB', dashboard_address=9002)


spotifyData = dd.read_csv('*AndUnique_allFeatures.csv', assume_missing=True)



#%% Summary data for reduced KMEANS

'''
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

# Now kmeans this?

hourAvgData = pd.read_pickle('hourAvgDataPickle')
hourAvgValues = hourAvgData.values

# Now scale?
# should use standardscaler so that we can go back to original data from
# the cluster centroids

scaler = preprocessing.StandardScaler()
scaler.fit(hourAvgValues)

hourAvgScaled = scaler.transform(hourAvgValues)

# also now quickly testing on just the means

hourAvgScaled = hourAvgScaled[:,[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28]]






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
# Now 5 seems to be the first "elbow"

# now k-means

skm_2 = dask_ml.cluster.KMeans(n_clusters=2, init_max_iter=2, oversampling_factor=2, max_iter=100)
skm_3 = dask_ml.cluster.KMeans(n_clusters=3, init_max_iter=3, oversampling_factor=2, max_iter=100)
skm_4 = dask_ml.cluster.KMeans(n_clusters=4, init_max_iter=4, oversampling_factor=2, max_iter=100)
skm_5 = dask_ml.cluster.KMeans(n_clusters=5, init_max_iter=5, oversampling_factor=2, max_iter=100)
skm_6 = dask_ml.cluster.KMeans(n_clusters=6, init_max_iter=6, oversampling_factor=2, max_iter=100)
skm_7 = dask_ml.cluster.KMeans(n_clusters=7, init_max_iter=7, oversampling_factor=2, max_iter=100)
skm_8 = dask_ml.cluster.KMeans(n_clusters=8, init_max_iter=8, oversampling_factor=2, max_iter=100)


skm_2.fit(hourAvgScaled)
skm_3.fit(hourAvgScaled)
skm_4.fit(hourAvgScaled)
skm_5.fit(hourAvgScaled)
skm_6.fit(hourAvgScaled)
skm_7.fit(hourAvgScaled)
skm_8.fit(hourAvgScaled)


skm_2_labels = skm_2.labels_.compute()
skm_3_labels = skm_3.labels_.compute()
skm_4_labels = skm_4.labels_.compute()
skm_5_labels = skm_5.labels_.compute()
skm_6_labels = skm_6.labels_.compute()
skm_7_labels = skm_7.labels_.compute()
skm_8_labels = skm_8.labels_.compute()



kmeansData = {'hour': np.arange(0,168),
			  'k2': skm_2_labels,
			  'k3': skm_3_labels,
			  'k4': skm_4_labels,
			  'k5': skm_5_labels,
			  'k6': skm_6_labels,
			  'k7': skm_7_labels,
			  'k8': skm_8_labels  
			  }
kmeansLabel = pd.DataFrame(kmeansData, columns = ['k2', 'k3', 
											   'k4', 'k5', 'k6',
											   'k7', 'k8'])


# now for a plot
# got to normalize first

#kmeansLabel.to_pickle('kmeansLabels')


kmeansNorm = (kmeansLabel-kmeansLabel.mean())/kmeansLabel.std()
plot = sns.heatmap(kmeansNorm[['k2', 'k3', 'k4', 'k5', 'k6', 'k7']], annot=False)

plot7 = sns.heatmap(kmeansLabel[['k5']])


#%% now save the cluster centroids
# save in transformed mode
clusterCentroids = {'k2' : scaler.inverse_transform(skm_2.cluster_centers_),
					'k3' : scaler.inverse_transform(skm_3.cluster_centers_),
					'k4' : scaler.inverse_transform(skm_4.cluster_centers_),
					'k5' : scaler.inverse_transform(skm_5.cluster_centers_),
					'k6' : scaler.inverse_transform(skm_6.cluster_centers_),
					'k7' : scaler.inverse_transform(skm_7.cluster_centers_),
					'k8' : scaler.inverse_transform(skm_8.cluster_centers_),}

cc2 = pd.DataFrame(scaler.inverse_transform(skm_2.cluster_centers_), columns=['bsMean', 'bsStd',
	'loudMean', 'loudStd',
	'mechMean', 'mechStd',
	'orgMean', 'orgStd',
	'tempMean', 'tempStd',
	'acousMean', 'acousStd',
	'bouncMean', 'bouncStd',
	'danceMean', 'danceStd',
	'dynMean', 'dynStd',
	'enerMean', 'enerStd',
	'flatMean', 'flatStd',
	'instMean', 'instStd',
	'liveMean', 'liveStd',
	'speechMean', 'speechStd',
	'valMean', 'valStd'])
cc2.to_pickle('clusterCentroids_k2')

cc3 = pd.DataFrame(scaler.inverse_transform(skm_3.cluster_centers_), columns=['bsMean', 'bsStd',
	'loudMean', 'loudStd',
	'mechMean', 'mechStd',
	'orgMean', 'orgStd',
	'tempMean', 'tempStd',
	'acousMean', 'acousStd',
	'bouncMean', 'bouncStd',
	'danceMean', 'danceStd',
	'dynMean', 'dynStd',
	'enerMean', 'enerStd',
	'flatMean', 'flatStd',
	'instMean', 'instStd',
	'liveMean', 'liveStd',
	'speechMean', 'speechStd',
	'valMean', 'valStd'])
cc3.to_pickle('clusterCentroids_k3')

cc4 = pd.DataFrame(scaler.inverse_transform(skm_4.cluster_centers_), columns=['bsMean', 'bsStd',
	'loudMean', 'loudStd',
	'mechMean', 'mechStd',
	'orgMean', 'orgStd',
	'tempMean', 'tempStd',
	'acousMean', 'acousStd',
	'bouncMean', 'bouncStd',
	'danceMean', 'danceStd',
	'dynMean', 'dynStd',
	'enerMean', 'enerStd',
	'flatMean', 'flatStd',
	'instMean', 'instStd',
	'liveMean', 'liveStd',
	'speechMean', 'speechStd',
	'valMean', 'valStd'])
cc4.to_pickle('clusterCentroids_k4')

cc5 = pd.DataFrame(scaler.inverse_transform(skm_5.cluster_centers_), columns=['bsMean', 'bsStd',
	'loudMean', 'loudStd',
	'mechMean', 'mechStd',
	'orgMean', 'orgStd',
	'tempMean', 'tempStd',
	'acousMean', 'acousStd',
	'bouncMean', 'bouncStd',
	'danceMean', 'danceStd',
	'dynMean', 'dynStd',
	'enerMean', 'enerStd',
	'flatMean', 'flatStd',
	'instMean', 'instStd',
	'liveMean', 'liveStd',
	'speechMean', 'speechStd',
	'valMean', 'valStd'])
cc5.to_pickle('clusterCentroids_k5')










'''

hourAvgData.index = hourAvgData.index.set_names([
	'bsMean', 'bsStd',
	'loudMean', 'loudStd',
	'mechMean', 'mechStd',
	'orgMean', 'orgStd',
	'tempMean', 'tempStd',
	'acousMean', 'acousStd',
	'bouncMean', 'bouncStd',
	'danceMean', 'danceStd',
	'dynMean', 'dynStd',
	'enerMean', 'enerStd',
	'flatMean', 'flatStd',
	'instMean', 'instStd',
	'liveMean', 'liveStd',
	'speechMean', 'speechStd',
	'valMean', 'valStd'])
hourAvgData.reset_index(inplace=True)


'''


#%%
# just plot together for fun. Need to melt it though.
scaledDataFrame = pd.DataFrame(hourAvgScaled, index=range(0,168), columns=['bsMean', 'bsStd',
	'loudMean', 'loudStd',
	'mechMean', 'mechStd',
	'orgMean', 'orgStd',
	'tempMean', 'tempStd',
	'acousMean', 'acousStd',
	'bouncMean', 'bouncStd',
	'danceMean', 'danceStd',
	'dynMean', 'dynStd',
	'enerMean', 'enerStd',
	'flatMean', 'flatStd',
	'instMean', 'instStd',
	'liveMean', 'liveStd',
	'speechMean', 'speechStd',
	'valMean', 'valStd'])
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
	
	
# beat strength only
plt.figure(figsize=(22,8))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['mechMean', 'orgMean', 'hour']], ['hour']),
													   label=None)
allPlot.legend_.remove()
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('K-means clustering of MIR-data, k=5')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
	









#%% KMEANS on all the data
'''
km_2 = dask_ml.cluster.KMeans(n_clusters=2, init_max_iter=2, oversampling_factor=2, max_iter=20)

km_2.fit(spotifyData[['beat_strength', 'mechanism', 'organism', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']])


these_centers = km_2.cluster_centers_
np.savetxt('2centers.csv', these_centers, delimiter=',', fmt='%10.6f')

hours = spotifyData[['unique', 'hour_of_day']]
hours = hours.assign(kLabel = km_2.labels_)
hours.to_csv('/kmeansResults/2Clusters_*.csv', single_file=False)



# Should try to plot things up a bit here. But first, need some sort of summary
# statistics from the file's we're writing




cluster2data = dd.read_csv('/kmeansResults/2Clusters_*.csv', assume_missing=True)


uniqueHourData = cluster2data.groupby('unique').agg({'kLabel' : ['value_counts']}).compute()



'''


















