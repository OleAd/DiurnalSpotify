# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:09:18 2021

@author: olehe

This script does detailed descriptive statistics + distributions calculations



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
client = Client(n_workers=1, threads_per_worker=7, processes=False, memory_limit='28GB', dashboard_address=9002)

#%%
labels=pd.read_pickle('data/kmeansLabelsForKequal5')

spotifyData = dd.read_csv('C:/Users/olehe/Desktop/MIB/Projects/Spotify_2019_2020/SpotifyData/*AndUnique_allFeatures.csv', assume_missing=True)

# first, need to groupby the actual data-subdivisions
# Note that these subdivision have different labels than what is used in the manuscript
subMorning = labels.index[labels[0]==4].tolist()
subMidnight = labels.index[labels[0]==2].tolist()
subAfternoon = labels.index[labels[0]==1].tolist()
subEvening = labels.index[labels[0]==3].tolist()
subEarlyMorning = labels.index[labels[0]==0].tolist()



#%% Not doable, simply too heavy calculations.

#morningSkew = spotifyData[spotifyData['unique'].isin(subMorning)][['danceability',
																	  'bounciness',
																	  'beat_strength',
																	  'mechanism',
																	  'organism',
																	  'dyn_range_mean',
																	  'flatness',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()

#morningSkewValue = stats.kurtosis(morningSkew['danceability'])



# Need to do this per subdivision for disk space reasons.

subMorningData = spotifyData[spotifyData['unique'].isin(subMorning)][['danceability',
																	  'bounciness',
																	  'beat_strength',
																	  'mechanism',
																	  'organism',
																	  'dyn_range_mean',
																	  'flatness',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()

subMorningData.to_csv('MorningDataDistribution_full.csv')



subMidnightData = spotifyData[spotifyData['unique'].isin(subMidnight)][['danceability',
																	  'bounciness',
																	  'beat_strength',
																	  'mechanism',
																	  'organism',
																	  'dyn_range_mean',
																	  'flatness',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()
subMidnightData.to_csv('MidnightDataDistribution_full.csv')

subAfternoonData = spotifyData[spotifyData['unique'].isin(subAfternoon)][['danceability',
																	  'bounciness',
																	  'beat_strength',
																	  'mechanism',
																	  'organism',
																	  'dyn_range_mean',
																	  'flatness',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()
subAfternoonData.to_csv('AfternoonDataDistribution_full.csv')


#still need to do evening
subEveningData = spotifyData[spotifyData['unique'].isin(subEvening)][['danceability',
																	  'bounciness',
																	  'beat_strength',
																	  'mechanism',
																	  'organism',
																	  'dyn_range_mean',
																	  'flatness',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()
subEveningData.to_csv('EveningDataDistribution.csv_full')



#early morning seems ok.
subEarlyMorningData = spotifyData[spotifyData['unique'].isin(subEarlyMorning)][['danceability',
																	  'bounciness',
																	  'beat_strength',
																	  'mechanism',
																	  'organism',
																	  'dyn_range_mean',
																	  'flatness',
																	 'energy',
																	 'loudness',
																	 'speechiness',
																	 'acousticness',
																	 'instrumentalness',
																	 'liveness',
																	 'valence',
																	 'tempo' ]].describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).compute()

subEarlyMorningData.to_csv('EarlyMorningDataDistribution_full.csv')



client.close()

#%% Plotting For 5 % and 95 %
distData_Afternoon = pd.read_csv('data/descriptivestats/AfternoonDataDistribution_full.csv', index_col=0)
distData_Evening = pd.read_csv('data/descriptivestats/EveningDataDistribution_full.csv', index_col=0)
distData_Midnight = pd.read_csv('data/descriptivestats/MidnightDataDistribution_full.csv', index_col=0)
distData_EarlyMorning = pd.read_csv('data/descriptivestats/EarlyMorningDataDistribution_full.csv', index_col=0)
distData_Morning = pd.read_csv('data/descriptivestats/MorningDataDistribution_full.csv', index_col=0)


five_Afternoon = pd.DataFrame(distData_Afternoon.loc['5%']).rename(columns={'5%':'Afternoon'})
five_Evening = pd.DataFrame(distData_Evening.loc['5%']).rename(columns={'5%':'Evening'})
five_Midnight = pd.DataFrame(distData_Midnight.loc['5%']).rename(columns={'5%':'Midnight'})
five_EarlyMorning = pd.DataFrame(distData_EarlyMorning.loc['5%']).rename(columns={'5%':'EarlyMorning'})
five_Morning = pd.DataFrame(distData_Morning.loc['5%']).rename(columns={'5%':'Morning'})

five_all = pd.concat([five_Morning,
					  five_Afternoon,
					  five_Evening,
					  five_Midnight,
					  five_EarlyMorning], axis=1)

ninetyfive_Afternoon = pd.DataFrame(distData_Afternoon.loc['95%']).rename(columns={'95%':'Afternoon'})
ninetyfive_Evening = pd.DataFrame(distData_Evening.loc['95%']).rename(columns={'95%':'Evening'})
ninetyfive_Midnight = pd.DataFrame(distData_Midnight.loc['95%']).rename(columns={'95%':'Midnight'})
ninetyfive_EarlyMorning = pd.DataFrame(distData_EarlyMorning.loc['95%']).rename(columns={'95%':'EarlyMorning'})
ninetyfive_Morning = pd.DataFrame(distData_Morning.loc['95%']).rename(columns={'95%':'Morning'})

ninetyfive_all = pd.concat([ninetyfive_Morning,
					  ninetyfive_Afternoon,
					  ninetyfive_Evening,
					  ninetyfive_Midnight,
					  ninetyfive_EarlyMorning], axis=1)



danceability_dist = pd.concat([
	pd.DataFrame(five_all.loc['dyn_range_mean']).rename(columns={'dyn_range_mean':'5%'}),
	pd.DataFrame(ninetyfive_all.loc['dyn_range_mean']).rename(columns={'dyn_range_mean':'95%'})], axis=1)

sns.lineplot(data=danceability_dist, sort=False)


# trying to subtract to get a sense of width of distribution.

diff_dist = ninetyfive_all-five_all

# rename column
diff_dist.rename(columns={'EarlyMorning':'Late Night/Early Morning'}, inplace=True)



sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['danceability'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Danceability')
plt.savefig('plots/DistWidt_danceability.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['bounciness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Bounciness')
plt.savefig('plots/DistWidt_bounciness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['beat_strength'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Beat Strength')
plt.savefig('plots/DistWidt_beatstrength.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['mechanism'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Mechanism')
plt.savefig('plots/DistWidt_mechanism.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['organism'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Organism')
plt.savefig('plots/DistWidt_organism.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['dyn_range_mean'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Dynamics')
plt.savefig('plots/DistWidt_dynamics.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['flatness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Flatness')
plt.savefig('plots/DistWidt_flatness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['energy'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Energy')
plt.savefig('plots/DistWidt_energy.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['loudness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Loudness')
plt.savefig('plots/DistWidt_loudness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['speechiness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Speechiness')
plt.savefig('plots/DistWidt_speechiness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['acousticness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Acousticness')
plt.savefig('plots/DistWidt_acousticness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['instrumentalness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Instrumentalness')
plt.savefig('plots/DistWidt_instrumentalness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['liveness'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Liveness')
plt.savefig('plots/DistWidt_liveness.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['valence'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Valence')
plt.savefig('plots/DistWidt_valence.svg', format='svg')

sns.set(style='whitegrid')
plt.figure(figsize=(6,4))
thisPlot = sns.lineplot(data=diff_dist.loc['tempo'], sort=False)
thisPlot.set(ylabel='Distribution width', title='Tempo')
plt.savefig('plots/DistWidt_tempo.svg', format='svg')




