# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 10:38:11 2020

@author: olehe


This script downloads samples from Spotify according to the subdivision of the day
"""

#%% Do imports
import sys
import spotipy
import spotipy.util as util
import pandas as pd
import time
import random
import os.path
import os
import requests
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#%% Initialise Spotipy
# This initiates a token

token=util.prompt_for_user_token('HIDDEN', scope=None,
						   client_id='HIDDEN',
						   client_secret='HIDDEN',
						   redirect_uri='HIDDEN')
sp = spotipy.Spotify(auth=token)


#%% Load database


# only scraped
allFeaturesDatabase = pd.read_csv('datasets/dataset_for_individual_Tracks.csv', low_memory=False)
allFeaturesDatabase.drop(columns='Unnamed: 0.1', inplace=True)

# get rid of some of the rain/whitenoise 
rain=allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Rain')]
rain = rain[rain['loudness'] <= -20]
rain = rain[rain['danceability'] < 0.4]
allFeaturesDatabase.drop(rain.index, inplace=True)

# get rid of noise tracks
noise = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Noise')]
allFeaturesDatabase.drop(noise.index, inplace=True)


# get rid of zero tempo and first pass of audiobooks
zeroTempo = allFeaturesDatabase[allFeaturesDatabase['tempo'] == 0]
stories = allFeaturesDatabase[allFeaturesDatabase['speechiness'] >= 0.94]
allFeaturesDatabase.drop(zeroTempo.index, inplace=True)
allFeaturesDatabase.drop(stories.index, inplace=True)

# drop some specific audiobooks
kat = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('schwarze Kat')]
auge = allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Im Auge des')]

# and another round of removing rain
rain=allFeaturesDatabase[allFeaturesDatabase['track_name'].str.contains('Rain')]
# take them by the index
rainIndex = rain[rain['Unnamed: 0'] > 159000]
rainIndex = rainIndex[rainIndex['Unnamed: 0'] < 161500]
allFeaturesDatabase.drop(rainIndex.index, inplace=True)
allFeaturesDatabase.drop(kat.index, inplace=True)
allFeaturesDatabase.drop(auge.index, inplace=True)

# and some more audio books

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


#%% Initiate a dataframe to hold just the chosen songs

chosenTracks = pd.DataFrame(columns=['track_name', 'track_id', 'SampleURL',
									 'danceability', 'energy', 'loudness',
									 'speechiness', 'acousticness', 'instrumentalness',
									 'acousticness', 'instrumentalness',
									 'liveness', 'valence', 'tempo', 'subdivision'])

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

em_P_data = ([em_P.values[1,1]-(em_P.values[2,1]*0.1),
			  em_P.values[1,2]-(em_P.values[2,2]*0.1),
			  em_P.values[1,3],
			  em_P.values[1,4],
			  em_P.values[1,5],
			  em_P.values[1,6],
			  em_P.values[1,7]+(em_P.values[2,7]*0.1),
			  em_P.values[1,8]-(em_P.values[2,8]*0.1),
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

a_P_data = ([a_P.values[1,1]+(a_P.values[2,1]*0.1),
			  a_P.values[1,2]+(a_P.values[2,2]*0.1),
			  a_P.values[1,3],
			  a_P.values[1,4],
			  a_P.values[1,5],
			  a_P.values[1,6],
			  a_P.values[1,7]-(a_P.values[2,7]*0.1),
			  a_P.values[1,8]+(a_P.values[2,8]*0.1),
			  a_P.values[1,9]])

a_P_values = pd.Series(a_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])


a_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] >= a_P_values['danceability']) &
	(allFeaturesDatabase['energy'] >= a_P_values['energy']) &
	(allFeaturesDatabase['loudness'] >= a_P_values['loudness']) &
	(allFeaturesDatabase['liveness'] <= a_P_values['liveness']) &
	(allFeaturesDatabase['valence'] >= a_P_values['valence']) &
	(allFeaturesDatabase['tempo'] >= a_P_values['tempo'])]


# now, here we get a lot of hits. Which to choose? Maybe just random?
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

# changing now to mean +- half std

m_P_data = ([m_P.values[1,1]-(m_P.values[2,1]*0.1),
			  m_P.values[1,2]+(m_P.values[2,2]*0.1),
			  m_P.values[1,3],
			  m_P.values[1,4],
			  m_P.values[1,5],
			  m_P.values[1,6],
			  m_P.values[1,7]+(m_P.values[2,7]*0.1),
			  m_P.values[1,8]+(m_P.values[2,8]*0.1),
			  m_P.values[1,9]])


m_P_values = pd.Series(m_P_data, index=['danceability', 'energy', 'loudness', 
								   'speechiness', 'acousticness', 'instrumentalness',
								   'liveness', 'valence', 'tempo'])


m_Tracks = allFeaturesDatabase[
	(allFeaturesDatabase['danceability'] <= m_P_values['danceability']) &
	(allFeaturesDatabase['energy'] >= m_P_values['energy']) &
	(allFeaturesDatabase['loudness'] >= m_P_values['loudness']) &
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

mn_P_data = ([mn_P.values[1,1]-(mn_P.values[2,1]*0.1),
			  mn_P.values[1,2]+(mn_P.values[2,2]*0.1),
			  mn_P.values[1,3],
			  mn_P.values[1,4],
			  mn_P.values[1,5],
			  mn_P.values[1,6],
			  mn_P.values[1,7]+(mn_P.values[2,7]*0.1),
			  mn_P.values[1,8]-(mn_P.values[2,8]*0.1),
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
#%% for evening
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


e_P_data = ([e_P.values[1,1]+(e_P.values[2,1]*0.1),
			  e_P.values[1,2]-(e_P.values[2,2]*0.1),
			  e_P.values[1,3],
			  e_P.values[1,4],
			  e_P.values[1,5],
			  e_P.values[1,6],
			  e_P.values[1,7]-(e_P.values[2,7]*0.1),
			  e_P.values[1,8]-(e_P.values[2,8]*0.1),
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
#%% Download tracks


# for early morning
countTracks=0
em_Tracks.reset_index(inplace=True, drop=True)
for ind in em_Tracks.index:
	track = em_Tracks.iloc[ind]
	thisFolder = 'earlyMorning'
	thisID = track[2]
	thisName = track[1]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		countTracks += 1
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('Unnamed: 0')
			# add one index to account for subdivision
			toDataFrame.at['subdivision'] = 'earlyMorning'
			# add to big dataframe
			chosenTracks = chosenTracks.append(toDataFrame, ignore_index=True)
			
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)				
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for early morning\n\n')

# for midnight
countTracks = 0
mn_Tracks.reset_index(inplace=True, drop=True)
for ind in mn_Tracks.index:
	track = mn_Tracks.iloc[ind]
	thisFolder = 'midnight'
	thisID = track[2]
	thisName = track[1]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('Unnamed: 0')
			# add one index to account for subdivision
			toDataFrame.at['subdivision'] = 'midnight'
			# add to big dataframe
			chosenTracks = chosenTracks.append(toDataFrame)
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
	
print('\n\nFinished for midnight\n\n')
# for morning
countTracks = 0
m_Tracks.reset_index(inplace=True, drop=True)
for ind in m_Tracks.index:
	track = m_Tracks.iloc[ind]
	thisFolder = 'morning'
	thisID = track[2]
	thisName = track[1]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		time.sleep(0.05)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('Unnamed: 0')
			# add one index to account for subdivision
			toDataFrame.at['subdivision'] = 'morning'
			# add to big dataframe
			chosenTracks = chosenTracks.append(toDataFrame)
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for morning\n\n')


# for afternoon
countTracks = 0
a_Tracks.reset_index(inplace=True, drop=True)
for ind in a_Tracks.index:
	track = a_Tracks.iloc[ind]
	thisFolder = 'afternoon'
	thisID = track[2]
	thisName = track[1]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('Unnamed: 0')
			# add one index to account for subdivision
			toDataFrame.at['subdivision'] = 'afternoon'
			# add to big dataframe
			chosenTracks = chosenTracks.append(toDataFrame)
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)
				time.sleep(0.1)
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')	
	if countTracks >= 500:
		break
print('\n\nFinished for afternoon\n\n')


# for evening
countTracks = 0
e_Tracks.reset_index(inplace=True, drop=True)
for ind in e_Tracks.index:
	track = e_Tracks.iloc[ind]
	thisFolder = 'evening'
	thisID = track[2]
	thisName = track[1]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('Unnamed: 0')
			# add one index to account for subdivision
			toDataFrame.at['subdivision'] = 'evening'
			# add to big dataframe
			chosenTracks = chosenTracks.append(toDataFrame)
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)
				time.sleep(0.1)
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for evening\n\n')


#%%	Inspect the chosen songs


# uncomment if you want to save again
#chosenTracks.drop_duplicates(subset=['track_name'], inplace=True)
#chosenTracks.to_pickle('chosenTracks.pkl')

# Read in the chosen tracks
chosenTracks = pd.read_pickle('data/chosenTracks.pkl')

# First, just look at all of them.

summaryStats = chosenTracks.groupby('subdivision').describe()

# Divide
emStats = chosenTracks.loc[chosenTracks['subdivision'] == 'earlyMorning']
mStats = chosenTracks.loc[chosenTracks['subdivision'] == 'morning']
aStats = chosenTracks.loc[chosenTracks['subdivision'] == 'afternoon']
eStats = chosenTracks.loc[chosenTracks['subdivision'] == 'evening']
mnStats = chosenTracks.loc[chosenTracks['subdivision'] == 'midnight']

# Plots for some stats
sns.distplot(emStats[['valence']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mStats[['valence']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(aStats[['valence']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(eStats[['valence']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mnStats[['valence']], kde=True, hist=False,  rug=False, norm_hist=True)
plt.xlim(0,1)

sns.distplot(emStats[['danceability']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mStats[['danceability']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(aStats[['danceability']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(eStats[['danceability']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mnStats[['danceability']], kde=True, hist=False,  rug=False, norm_hist=True)
plt.xlim(0,1)

sns.distplot(emStats[['energy']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mStats[['energy']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(aStats[['energy']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(eStats[['energy']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mnStats[['energy']], kde=True, hist=False,  rug=False, norm_hist=True)
plt.xlim(0,1)


sns.distplot(emStats[['tempo']], hist=False, rug=False)
sns.distplot(mStats[['tempo']], hist=False, rug=False)
sns.distplot(aStats[['tempo']], hist=False, rug=False)
sns.distplot(eStats[['tempo']], hist=False, rug=False)
sns.distplot(mnStats[['tempo']], hist=False, rug=False)
plt.xlim(30,200)

sns.distplot(emStats[['liveness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mStats[['liveness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(aStats[['liveness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(eStats[['liveness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mnStats[['liveness']], kde=True, hist=False,  rug=False, norm_hist=True)
plt.xlim(0,1)

sns.distplot(emStats[['loudness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mStats[['loudness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(aStats[['loudness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(eStats[['loudness']], kde=True, hist=False, rug=False, norm_hist=True)
sns.distplot(mnStats[['loudness']], kde=True, hist=False,  rug=False, norm_hist=True)
plt.xlim(-35,0)




#%% PCA plot
# Look at the 2 first pca components of the chosen data

chosenValues = chosenTracks[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]
chosenLabels = chosenTracks['subdivision']
mapping = {'earlyMorning': 0,
		   'morning': 1,
		   'afternoon': 2,
		   'evening': 3,
		   'midnight': 4}
chosenInts = chosenLabels.map(lambda s: mapping.get(s) if s in mapping else s)


standardChosenValues = StandardScaler().fit_transform(chosenValues)
chosenPca = PCA(n_components=2, whiten=True)
chosenPca.fit(standardChosenValues)
X = chosenPca.transform(standardChosenValues)

target_ids = range(len(chosenLabels))

labels = chosenLabels.unique()

plt.figure(1, figsize=(8,6))

for i, c, label in zip(target_ids, 'rgbcmykw', labels):
	plt.scatter(X[chosenInts == i, 0], X[chosenInts == i, 1],
			 c=c, label=label)

ax = plt.gca()
#ax.set_xlim((-0.15,0.15))
plt.legend()
plt.show()


# also, check the factor loadings.

weightMatrixChosen = chosenPca.components_.T * np.sqrt(chosenPca.explained_variance_)



#%% Reducing candidate tracks by PCA
# Here, take a PCA of the features for each subdivision, choose tracks on the
# extremes of the spectrum.



# early morning
emValues = emStats[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]
emValues = StandardScaler().fit_transform(emValues)


pca = PCA(n_components=1)
pca.fit(emValues)
emTrans = pca.transform(emValues)
emStats['pca'] = emTrans.tolist()
emStatsSorted = emStats.sort_values(by=['pca'])
midValue = round(len(emStatsSorted)*0.5)
emSelection = emStatsSorted.iloc[[-1, -2, -3, -4, -5, midValue-2, midValue-1, midValue,
								  midValue+1, midValue+2, 0, 1, 2, 3, 4]]



# morning
mValues = mStats[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]

mValues = StandardScaler().fit_transform(mValues)

pca = PCA(n_components=1)
pca.fit(mValues)
mTrans = pca.transform(mValues)
mStats['pca'] = mTrans.tolist()
mStatsSorted = mStats.sort_values(by=['pca'])
midValue = round(len(mStatsSorted)*0.5)
mSelection = mStatsSorted.iloc[[-1, -2, -3, -4, -5, midValue-2, midValue-1, midValue,
								  midValue+1, midValue+2, 0, 1, 2, 3, 4]]
# remove a duplicate here



# afternoon
aValues = aStats[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]

aValues = StandardScaler().fit_transform(aValues)


pca = PCA(n_components=1)
pca.fit(aValues)
aTrans = pca.transform(aValues)
aStats['pca'] = aTrans.tolist()
aStatsSorted = aStats.sort_values(by=['pca'])
midValue = round(len(aStatsSorted)*0.5)
aSelection = aStatsSorted.iloc[[-1, -2, -3, -4, -5, midValue-2, midValue-1, midValue,
								  midValue+1, midValue+2, 0, 1, 2, 3, 4]]

# evening
eValues = eStats[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]
eValues = StandardScaler().fit_transform(eValues)

pca = PCA(n_components=1)
pca.fit(aValues)
eTrans = pca.transform(eValues)
eStats['pca'] = eTrans.tolist()
eStatsSorted = eStats.sort_values(by=['pca'])
midValue = round(len(eStatsSorted)*0.5)
eSelection = eStatsSorted.iloc[[-1, -2, -3, -4, -5, midValue-2, midValue-1, midValue,
								  midValue+1, midValue+2, 0, 1, 2, 3, 4]]

# midnight

mnValues = mnStats[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]
mnValues = StandardScaler().fit_transform(mnValues)

pca = PCA(n_components=1)
pca.fit(mnValues)
mnTrans = pca.transform(mnValues)
mnStats['pca'] = mnTrans.tolist()
mnStatsSorted = mnStats.sort_values(by=['pca'])
midValue = round(len(mnStatsSorted)*0.5)
mnSelection = mnStatsSorted.iloc[[-1, -2, -3, -4, -5, midValue-2, midValue-1, midValue,
								  midValue+1, midValue+2, 0, 1, 2, 3, 4]]



#%% Download and analyse the new selection of tracks
# Need to re-download these tracks

selectionTracks = pd.DataFrame(columns=['track_name', 'track_id', 'SampleURL',
									 'danceability', 'energy', 'loudness',
									 'speechiness', 'acousticness', 'instrumentalness',
									 'acousticness', 'instrumentalness',
									 'liveness', 'valence', 'tempo', 'subdivision'])


# for early morning
countTracks=0
emSelection.reset_index(inplace=True, drop=True)
for ind in emSelection.index:
	track = emSelection.iloc[ind]
	thisFolder = 'earlyMorning/selection'
	thisID = track[1]
	thisName = track[0]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		countTracks += 1
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('pca')
			
			# add to big dataframe
			selectionTracks = selectionTracks.append(toDataFrame, ignore_index=True)
			
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)				
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for early morning\n\n')

# for  morning
countTracks=0
mSelection.reset_index(inplace=True, drop=True)
for ind in mSelection.index:
	track = mSelection.iloc[ind]
	thisFolder = 'morning/selection'
	thisID = track[1]
	thisName = track[0]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		countTracks += 1
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('pca')
			
			# add to big dataframe
			selectionTracks = selectionTracks.append(toDataFrame, ignore_index=True)
			
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)				
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for morning\n\n')

# for afternoon
countTracks=0
aSelection.reset_index(inplace=True, drop=True)
for ind in aSelection.index:
	track = aSelection.iloc[ind]
	thisFolder = 'afternoon/selection'
	thisID = track[1]
	thisName = track[0]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		countTracks += 1
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('pca')
			
			# add to big dataframe
			selectionTracks = selectionTracks.append(toDataFrame, ignore_index=True)
			
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)				
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for afternoon\n\n')

# for evening
countTracks=0
eSelection.reset_index(inplace=True, drop=True)
for ind in eSelection.index:
	track = eSelection.iloc[ind]
	thisFolder = 'evening/selection'
	thisID = track[1]
	thisName = track[0]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		countTracks += 1
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('pca')
			
			# add to big dataframe
			selectionTracks = selectionTracks.append(toDataFrame, ignore_index=True)
			
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)				
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for evening\n\n')


# for midnight
countTracks=0
mnSelection.reset_index(inplace=True, drop=True)
for ind in mnSelection.index:
	track = mnSelection.iloc[ind]
	thisFolder = 'midnight/selection'
	thisID = track[1]
	thisName = track[0]
	thisName = thisName.replace("/", "")
	thisName = thisName.replace("'", "")
	thisName = thisName[0:20]
	thisInfo = sp.track(thisID)
	thisUrl = thisInfo['preview_url']
	thisPopularity = thisInfo['popularity']
	if thisUrl != None:
		countTracks += 1
		time.sleep(0.3)
		# also need to check if we actually get any data
		thisSample = requests.get(thisUrl)
		if len(thisSample.content) > 0:
			#save this track to the dataframe of chosen songs
			toDataFrame = track.drop('pca')
			
			# add to big dataframe
			selectionTracks = selectionTracks.append(toDataFrame, ignore_index=True)
			
			
			saveName = thisFolder + '/' + str(thisPopularity) + ' Name - ' + thisName + ' ID - ' + thisID + '.mp3'
			saveName = saveName.replace(":", "")
			saveName = saveName.replace("!", "")
			saveName = saveName.replace('"', '')
			saveName = saveName.replace('?', '')
			saveName = saveName.replace('<', '')
			saveName = saveName.replace('>', '')
			saveName = saveName.replace('\\', '')
			saveName = saveName.replace('*', '')
			if os.path.isfile(saveName) == False:
				open(saveName, 'wb').write(thisSample.content)				
		else:
			print('One did not download successfully')
	else:
		print('Skipped one due to no sample available')
	if countTracks >= 500:
		break
print('\n\nFinished for midnight\n\n')



#%% Inspected the selected tracks

# Uncomment if you've done the section above, and want to save them
#selectionTracks.to_pickle('selectionTracks.pkl')
selectionTracks = pd.read_pickle('data/selectionTracks.pkl')

# Look at the 2 first pca components of the selected tracks

summaryStatsSelection = selectionTracks.groupby('subdivision').describe()

selectionValues = selectionTracks[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]

selectionValues = StandardScaler().fit_transform(selectionValues)

selectionLabels = selectionTracks['subdivision']
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

plt.figure(1, figsize=(8,6))

for i, c, label in zip(target_ids, 'rgbcmykw', labels):
	plt.scatter(X[selectionInts == i, 0], X[selectionInts == i, 1],
			 c=c, label=label)

plt.legend()
plt.show()




weightMatrixSelected = allPca.components_.T * np.sqrt(allPca.explained_variance_)



#%% Researcher's selection

furtherSelection = selectionTracks[selectionTracks['track_name']
								   .str
								   .contains('The Ocean|Running Up That Hill|Flow|Lake Oconee|Haul|Crybaby|Supreme|Ill Live On|Takin Ova|Backstage|Unfinished Sympathy|St Elmos Fire|Dive|Only Girl|We Got Love|Pepe|Bridge You Burn|Cardigan|I Mine Øjne|Unfreeze|YOU CANT HOLD MY HEA|Last to Leave|Love Galore|Silent Disco|Not Ok|The Soul - Trio Riot|FFS|Stopper opp|Distant Light|Porcelain')]



furtherSelectionValues = furtherSelection[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]

furtherSelectionValues = StandardScaler().fit_transform(furtherSelectionValues)

furtherSelectionLabels = furtherSelection['subdivision']
mapping = {'earlyMorning': 0,
		   'morning': 1,
		   'afternoon': 2,
		   'evening': 3,
		   'midnight': 4}
furtherSelectionInts = furtherSelectionLabels.map(lambda s: mapping.get(s) if s in mapping else s)

allPca = PCA(n_components=2, whiten=True)
allPca.fit(furtherSelectionValues)
X = allPca.transform(furtherSelectionValues)

target_ids = range(len(furtherSelectionLabels))

labels = furtherSelectionLabels.unique()

plt.figure(1, figsize=(8,6))

for i, c, label in zip(target_ids, 'rgbcmykw', labels):
	plt.scatter(X[furtherSelectionInts == i, 0], X[furtherSelectionInts == i, 1],
			 c=c, label=label)
ax = plt.gca()

txt = furtherSelection['track_name'].values
for i in range(len(X)):
	ax.annotate(txt[i], [X[i,0], X[i,1]])
plt.legend()
plt.show()


#%% Dropping after selection

toDrop = furtherSelection[furtherSelection['track_name']
						  .str
						  .contains('Bridge You Burn|Love Galore|Haul|FFS|Not Ok|Silent Disco|Crybaby'
							  )]

furtherSelection.drop(toDrop.index, inplace=True)



furtherSelectionValues = furtherSelection[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]

furtherSelectionValues = StandardScaler().fit_transform(furtherSelectionValues)

furtherSelectionLabels = furtherSelection['subdivision']
mapping = {'earlyMorning': 0,
		   'morning': 1,
		   'afternoon': 2,
		   'evening': 3,
		   'midnight': 4}
furtherSelectionInts = furtherSelectionLabels.map(lambda s: mapping.get(s) if s in mapping else s)

allPca = PCA(n_components=2, whiten=True)
allPca.fit(furtherSelectionValues)
X = allPca.transform(furtherSelectionValues)

target_ids = range(len(furtherSelectionLabels))

labels = furtherSelectionLabels.unique()

plt.figure(1, figsize=(8,6))

for i, c, label in zip(target_ids, 'rgbcmykw', labels):
	plt.scatter(X[furtherSelectionInts == i, 0], X[furtherSelectionInts == i, 1],
			 c=c, label=label)
ax = plt.gca()

txt = furtherSelection['track_name'].values
txt[2]='Running Up That Hill'
txt[11]='Only Girl'
txt[17]='The Soul'

text = [plt.text(X[i,0], X[i,1], txt[i], ha='center', va='center') for i in range(len(X))]
adjust_text(text)
plt.legend()
plt.show()
plt.savefig('plots/PCAselection.svg', format='svg')
















