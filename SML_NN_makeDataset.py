# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:32:18 2020

@author: olehe

This script makes the training dataset for the neural network
"""


#%% Do imports

import pandas as pd
import os.path
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


#%% Load dataset and clean

allFeaturesDatabase = pd.read_csv('datasets/dataset_for_neural_network.csv', low_memory=False)
#allFeaturesDatabase.drop(columns='Unnamed: 0.1', inplace=True)

# get rid of some of the rain/whitenoise
rain=allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Rain')]
rain = rain[rain['loudness'] <= -20]
rain = rain[rain['danceability'] < 0.4]
allFeaturesDatabase.drop(rain.index, inplace=True)


# Get rid of noise tracks
noise = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Noise')]
allFeaturesDatabase.drop(noise.index, inplace=True)

# Get rid of 0-tempo tracks and spoken words (usually audiobooks released as cd's)
zeroTempo = allFeaturesDatabase[allFeaturesDatabase['tempo'] == 0]
stories = allFeaturesDatabase[allFeaturesDatabase['speechiness'] >= 0.94]
allFeaturesDatabase.drop(zeroTempo.index, inplace=True)
allFeaturesDatabase.drop(stories.index, inplace=True)

# Drop some specific German audiobooks
kat = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('schwarze Kat')]
auge = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Im Auge des')]

# Another go at removing rain
rain=allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Rain')]
# take them by the index
rainIndex = rain[rain['Unnamed: 0'] > 159000]
rainIndex = rainIndex[rainIndex['Unnamed: 0'] < 161500]
allFeaturesDatabase.drop(rainIndex.index, inplace=True)
allFeaturesDatabase.drop(kat.index, inplace=True)
allFeaturesDatabase.drop(auge.index, inplace=True)

# And a few more audiobooks

folge = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Folge')]
openSkies = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Open Skies')]
thunder = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Window Thunder')]
allFeaturesDatabase.drop(folge.index, inplace=True)
allFeaturesDatabase.drop(openSkies.index, inplace=True)
allFeaturesDatabase.drop(thunder.index, inplace=True)

teil = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Teil')]
kapitel = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Kapitel')]
allFeaturesDatabase.drop(kapitel.index, inplace=True)
allFeaturesDatabase.drop(teil.index, inplace=True)



#%% Now do the track selection per subdisivions

#%% Early morning
em_P = pd.read_csv('data/descriptivestats/EarlyMorningDataDistribution.csv')

# for early morning:
#danceability: down
#energy: down
#loudness: down
#speechiness: up
#acousticness: up
#instrumentalness: up
#liveness: up
#valence:down
#tempo: down


em_P_data = ([em_P.values[1,1],
			  em_P.values[1,2],
			  em_P.values[1,3],
			  em_P.values[1,4],
			  em_P.values[1,5],
			  em_P.values[1,6],
			  em_P.values[1,7],
			  em_P.values[1,8],
			  em_P.values[1,9]])

em_P_values = pd.Series(em_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])

# now locate possible songs in the candidate list

em_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] <= em_P_values['danceability']) &
	(allFeaturesDatabase['energy'] <= em_P_values['energy']) &
	(allFeaturesDatabase['loudness'] <= em_P_values['loudness']) &
	(allFeaturesDatabase['liveness'] >= em_P_values['liveness']) &
	(allFeaturesDatabase['valence'] <= em_P_values['valence']) &
	(allFeaturesDatabase['tempo'] <= em_P_values['tempo'])]



print('Number of tracks: ' + str(len(em_Tracks)))




#%% Afternoon

a_P = pd.read_csv('data/descriptivestats/AfternoonDataDistribution.csv')

# for afternoon
#danceability: up
#energy: up
#loudness: up
#speechiness: down
#acousticness: down
#instrumentalness: down
#liveness: down
#valence:up
#tempo: up


a_P_data = ([a_P.values[1,1],
			  a_P.values[1,2],
			  a_P.values[1,3],
			  a_P.values[1,4],
			  a_P.values[1,5],
			  a_P.values[1,6],
			  a_P.values[1,7],
			  a_P.values[1,8],
			  a_P.values[1,9]])


a_P_values = pd.Series(a_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])

# now locate possible songs in the candidate list

a_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] >= a_P_values['danceability']) &
	(allFeaturesDatabase['energy'] >= a_P_values['energy']) &
	(allFeaturesDatabase['loudness'] >= a_P_values['loudness']) &
	(allFeaturesDatabase['liveness'] <= a_P_values['liveness']) &
	(allFeaturesDatabase['valence'] >= a_P_values['valence']) &
	(allFeaturesDatabase['tempo'] >= a_P_values['tempo'])]


print('Number of tracks: ' + str(len(a_Tracks)))
#%% Morning

m_P = pd.read_csv('data/descriptivestats/MorningDataDistribution.csv')

# for morning
#danceability: down
#energy: up
#loudness: up
#speechiness: down
#acousticness: down
#instrumentalness: down
#liveness: up
#valence:up
#tempo: down

m_P_data = ([m_P.values[1,1],
			  m_P.values[1,2],
			  m_P.values[1,3],
			  m_P.values[1,4],
			  m_P.values[1,5],
			  m_P.values[1,6],
			  m_P.values[1,7],
			  m_P.values[1,8],
			  m_P.values[1,9]])


m_P_values = pd.Series(m_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])


# now locate possible songs in the candidate list

m_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] <= m_P_values['danceability']) &
	(allFeaturesDatabase['energy'] >= m_P_values['energy']) &
	(allFeaturesDatabase['loudness'] >= m_P_values['loudness']) &
	#(allFeaturesDatabase['speechiness'] <= m_P_values['speechiness']) &
	#(allFeaturesDatabase['acousticness'] <= m_P_values['acousticness']) &
	#(allFeaturesDatabase['instrumentalness'] <= m_P_values['instrumentalness']) &
	(allFeaturesDatabase['liveness'] >= m_P_values['liveness']) &
	(allFeaturesDatabase['valence'] >= m_P_values['valence']) &
	(allFeaturesDatabase['tempo'] <= m_P_values['tempo'])]

print('Number of tracks: ' + str(len(m_Tracks)))
#%% Midnight

mn_P = pd.read_csv('data/descriptivestats/MidnightDataDistribution.csv')

# for midnight
#danceability: down
#energy: up
#loudness: down
#speechiness: up
#acousticness: up
#instrumentalness: up
#liveness: up
#valence:down
#tempo: down

mn_P_data = ([mn_P.values[1,1],
			  mn_P.values[1,2],
			  mn_P.values[1,3],
			  mn_P.values[1,4],
			  mn_P.values[1,5],
			  mn_P.values[1,6],
			  mn_P.values[1,7],
			  mn_P.values[1,8],
			  mn_P.values[1,9]])

mn_P_values = pd.Series(mn_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])

# now locate possible songs in the candidate list

mn_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] <= mn_P_values['danceability']) &
	(allFeaturesDatabase['energy'] >= mn_P_values['energy']) &
	(allFeaturesDatabase['loudness'] <= mn_P_values['loudness']) &
	(allFeaturesDatabase['liveness'] >= mn_P_values['liveness']) &
	(allFeaturesDatabase['valence'] <= mn_P_values['valence']) &
	(allFeaturesDatabase['tempo'] <= mn_P_values['tempo'])]


print('Number of tracks: ' + str(len(mn_Tracks)))
#%% Evening
#for evening
#danceability: up
#energy: down
#loudness: up
#speechiness: up
#acousticness: down
#instrumentalness:down
#liveness: down
#valence:down
#tempo: up

e_P = pd.read_csv('data/descriptivestats/EveningDataDistribution.csv')


e_P_data = ([e_P.values[1,1],
			  e_P.values[1,2],
			  e_P.values[1,3],
			  e_P.values[1,4],
			  e_P.values[1,5],
			  e_P.values[1,6],
			  e_P.values[1,7],
			  e_P.values[1,8],
			  e_P.values[1,9]])

e_P_values = pd.Series(e_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])

# now locate possible songs in the candidate list

e_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] >= e_P_values['danceability']) &
	(allFeaturesDatabase['energy'] <= e_P_values['energy']) &
	(allFeaturesDatabase['loudness'] >= e_P_values['loudness']) &
	(allFeaturesDatabase['liveness'] <= e_P_values['liveness']) &
	(allFeaturesDatabase['valence'] <= e_P_values['valence']) &
	(allFeaturesDatabase['tempo'] >= e_P_values['tempo'])]


print('Number of tracks: ' + str(len(e_Tracks)))

#%% Combine into one dataframe

'''
0 = early morning
1 = morning
2 = afternoon
3 = evening
4 = midnight

'''

em_Tracks['subdivision'] = 0
m_Tracks['subdivision'] = 1
a_Tracks['subdivision'] = 2
e_Tracks['subdivision'] = 3
mn_Tracks['subdivision'] = 4

em_Tracks.pop('Unnamed: 0')
m_Tracks.pop('Unnamed: 0')
a_Tracks.pop('Unnamed: 0')
e_Tracks.pop('Unnamed: 0')
mn_Tracks.pop('Unnamed: 0')


selectedTracks = pd.concat([em_Tracks, m_Tracks, a_Tracks, e_Tracks, mn_Tracks], 
						   join='inner', ignore_index=True)


#%% Visualize data

summaryStatsSelection = selectedTracks.groupby('subdivision').describe()

selectionValues = selectedTracks[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]

selectionValues = StandardScaler().fit_transform(selectionValues)

selectionLabels = selectedTracks['subdivision']
mapping = {'earlyMorning': 0,
		   'morning': 1,
		   'afternoon': 2,
		   'evening': 3,
		   'midnight': 4}
selectionInts = selectionLabels.map(lambda s: mapping.get(s) if s in mapping else s)

allPca = PCA(n_components=2, whiten=True)
allPca.fit(selectionValues)
X = allPca.transform(selectionValues)

target_ids = range(len(selectionLabels))

labels = selectionLabels.unique()
# Considering having this as a supplementary plot
plt.figure(1, figsize=(8,6))

for i, c, label in zip(target_ids, 'rgbcmykw', labels):
	plt.scatter(X[selectionInts == i, 0], X[selectionInts == i, 1],
			 c=c, alpha=0.25, label=label)

plt.legend()
plt.show()



#%% Make dataset


em_train, em_test = train_test_split(em_Tracks, test_size=0.2)
m_train, m_test = train_test_split(m_Tracks, test_size=0.2)
a_train, a_test = train_test_split(a_Tracks, test_size=0.2)
e_train, e_test = train_test_split(e_Tracks, test_size=0.2)
mn_train, mn_test = train_test_split(mn_Tracks, test_size=0.2)


trainDataset = pd.concat([em_train, m_train, a_train, e_train, mn_train])
testDataset = pd.concat([em_test, m_test, a_test, e_test, mn_test])



toSaveDatasetTrain = trainDataset[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo', 'subdivision']]

toSaveDatasetTest = testDataset[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo', 'subdivision']]


# calculate weightings for training dataset.

totalLength = len(toSaveDatasetTrain)

emWeight = len(toSaveDatasetTrain[toSaveDatasetTrain['subdivision']==0]) / totalLength
mWeight = len(toSaveDatasetTrain[toSaveDatasetTrain['subdivision']==1]) / totalLength
aWeight = len(toSaveDatasetTrain[toSaveDatasetTrain['subdivision']==2]) / totalLength
eWeight = len(toSaveDatasetTrain[toSaveDatasetTrain['subdivision']==3]) / totalLength
mnWeight = len(toSaveDatasetTrain[toSaveDatasetTrain['subdivision']==4]) / totalLength


class_weights = class_weight.compute_class_weight('balanced', np.unique(toSaveDatasetTrain['subdivision']), toSaveDatasetTrain['subdivision'])

# Uncomment to save
#np.save('data/trainingClassweights', class_weights)
#toSaveDatasetTrain.to_pickle('data/trainingDataset.pkl')
#toSaveDatasetTest.to_pickle('data/holdoutDataset.pkl')





