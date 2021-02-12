# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:17:07 2020

@author: olehe


This script makes plots
And does the correlation between ATUS ADI and Musical Variability

"""


#%% Imports

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats
import scipy.signal



#%% Read data

# This is the Spotify data averaged per unique hour
hourAvgData = pd.read_pickle('data/hourAvgDataPickle')
hourAvgValues = hourAvgData.values

# Scaled data
scaler = preprocessing.StandardScaler()
scaler.fit(hourAvgValues)
hourAvgScaled = scaler.transform(hourAvgValues)


# Read the k-means labels
skm_5_labels = pd.read_pickle('data/kmeansLabelsForKequal5')
skm_5_labels = skm_5_labels.values



#%% Section for looking at standard deviations, and for plotting day cycles

# Make a non-scaled dataframe
rawDataFrame = pd.DataFrame(hourAvgValues, index=range(0,168), columns=['bsMean', 'bsStd',
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
rawDataFrame['hour'] = range(0,168)

# Make a scaled dataframe
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


# Calculate mean values for each subdivision, as well as std's.
rawDataFrame['subdivision'] = skm_5_labels

# For practical reasons, remap the subdivision labels
subDict ={0:'Late Night/Early Morning',
		  1:'Afternoon',
		  2:'Night',
		  3:'Evening',
		  4:'Morning'}

rawDataFrame.replace({'subdivision':subDict}, inplace=True)

# Mean values
meanValues = rawDataFrame[['bsMean', 'loudMean', 
						     'mechMean','orgMean',
							 'tempMean', 'acousMean',
							 'bouncMean', 'danceMean',
							 'dynMean', 'enerMean',
							 'flatMean', 'instMean',
							 'liveMean', 'speechMean',
							 'valMean','subdivision']].groupby('subdivision').mean()
# Mean STD values
meanStdValues = rawDataFrame[['bsStd', 'loudStd', 
						     'mechStd','orgStd',
							 'tempStd', 'acousMean',
							 'bouncStd', 'danceStd',
							 'dynStd', 'enerStd',
							 'flatStd', 'instStd',
							 'liveStd', 'speechStd',
							 'valStd','subdivision']].groupby('subdivision').mean()


# Plot of STDs over the week.
plt.figure(figsize=(22,8))
stdPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['bsStd', 'loudStd', 
													  'mechStd','orgStd',
													  'tempStd', 'acousStd',
													  'bouncStd', 'danceStd',
													  'dynStd', 'enerStd',
													  'flatStd', 'instStd',
													  'liveStd', 'speechStd',
													  'valStd','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
stdPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
stdPlot.set(xlim=(0,167))
stdPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
stdPlot.set_xticklabels(np.tile([0,6,12,18], 7))
stdPlot.set_title('STDs over the week')
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	stdPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
#plt.savefig("plots/weekSTDs.svg", format="svg")	


#%% Groupyby subdivision and mean.
	
groupStds = rawDataFrame[['bsMean', 'bsStd',
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
	'valMean', 'valStd','subdivision']].groupby('subdivision').mean()

#%% Section for plotting std's and then averaging over 24 hours


# Add hour of the day to the dataframe
hourOfDay = np.tile(np.arange(0,24),7)
scaledDataFrame['hourOfDay']=hourOfDay
stdsByHour = scaledDataFrame[['bsStd', 'loudStd', 
							  'mechStd','orgStd',
							  'tempStd', 'acousStd',
							  'bouncStd', 'danceStd',
							  'dynStd', 'enerStd',
							  'flatStd', 'instStd',
							  'liveStd', 'speechStd',
							  'valStd','hourOfDay']].groupby(['hourOfDay']).mean()

stdsByHour['hourOfDay'] = np.arange(0,24)



# Plot of 24-hour audio feature variability
plt.figure(figsize=(8,8))
stdsHourPlot = sns.lineplot(x='hourOfDay', y='value', hue='variable',
							data=pd.melt(scaledDataFrame[['bsStd', 'loudStd', 
							  'mechStd','orgStd',
							  'tempStd', 'acousStd',
							  'bouncStd', 'danceStd',
							  'dynStd', 'enerStd',
							  'flatStd', 'instStd',
							  'liveStd', 'speechStd',
							  'valStd','hourOfDay']], id_vars='hourOfDay'),
							label=None)
stdsHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
stdsHourPlot.set(xlim=(0,23))
stdsHourPlot.set_xticks([0, 6, 12, 18])
stdsHourPlot.set_xticklabels([0,6,12,18])
stdsHourPlot.set_title('24-hour audio feature variability')
#plt.savefig("plots/variability24hour.svg", format="svg")	


#%% Correlate with atus 

diversityLine = pd.read_pickle('data/aidValues.pkl')

# Average stds per hour, for all of the variables
stddata=pd.melt(scaledDataFrame[['bsStd', 'loudStd', 
							  'mechStd','orgStd',
							  'tempStd', 'acousStd',
							  'bouncStd', 'danceStd',
							  'dynStd', 'enerStd',
							  'flatStd', 'instStd',
							  'liveStd', 'speechStd',
							  'valStd','hourOfDay']], id_vars='hourOfDay')
stdLine = stddata[['hourOfDay', 'value']].groupby('hourOfDay').mean()


normal1=scipy.stats.normaltest(diversityLine['Diversity'])
normal2=scipy.stats.normaltest(stdLine['value'])
corr = scipy.stats.pearsonr(diversityLine['Diversity'], stdLine['value'])

print('Correlation is ' + str(np.round(corr[0],3)) + ', p=' + str(np.round(corr[1],3)))





#%% Plot for single features average over the 24-hour
# Reorganizing to match the subdivisions, by adding a 6 hour offset.

scaledDataFrame['hourOfDayOffset'] = scaledDataFrame['hourOfDay'] + 6


plt.figure(figsize=(22,2))
tempHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['tempMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
tempHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
tempHourPlot.set(xlim=(6,29))
tempHourPlot.set(ylim=(-2.5,2.5))
tempHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
tempHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
tempHourPlot.set_title('Tempo')
#plt.savefig("plots/temp24hour.svg", format="svg")	


plt.figure(figsize=(22,2))
danceHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['danceMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
danceHourPlot.set(xlim=(6,29))
danceHourPlot.set(ylim=(-2.5,2.5))
danceHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
danceHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
danceHourPlot.set_title('Danceability')
#plt.savefig("plots/dance24hour.svg", format="svg")	

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['loudMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Loudness')
#plt.savefig("plots/loud24hour.svg", format="svg")


# per reviewer, add these plots for all the audio features.


plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['dynMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Dynamics')
plt.savefig("plots/dyn24hour.svg", format="svg")


plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['bsMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Beat Strength')
plt.savefig("plots/bs24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['mechMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Mechanism')
plt.savefig("plots/mech24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['orgMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Organism')
plt.savefig("plots/org24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['acousMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Acousticness')
plt.savefig("plots/acous24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['bouncMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Bounciness')
plt.savefig("plots/bounce24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['enerMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Energy')
plt.savefig("plots/energy24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['flatMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Flatness')
plt.savefig("plots/flat24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['instMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Instrumentalness')
plt.savefig("plots/inst24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['liveMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Liveness')
plt.savefig("plots/live24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['speechMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Speechiness')
plt.savefig("plots/speech24hour.svg", format="svg")

plt.figure(figsize=(22,2))
loudHourPlot = sns.lineplot(x='hourOfDayOffset', y='value',
							data=pd.melt(scaledDataFrame[['valMean', 'hourOfDayOffset']], id_vars='hourOfDayOffset'),
							label=None)
#danceHourPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
loudHourPlot.set(xlim=(6,29))
loudHourPlot.set(ylim=(-2.5,2.5))
loudHourPlot.set_xticks([6, 9, 12, 15, 18, 21, 24, 27])
loudHourPlot.set_xticklabels([6, 9, 12, 15, 18, 21, 0, 3])
loudHourPlot.set_title('Valence')
plt.savefig("plots/valence24hour.svg", format="svg")




#%% Plot of all mean values over the entire week

# Plot for all the means
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
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('K-means clustering of MIR-data, k=5')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
#plt.savefig("plots/allVariables.svg", format="svg")	


#%% Smaller plots
	
# Plot for danceability, but smaller
plt.figure(figsize=(22,4))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['danceMean','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('Danceability')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
plt.savefig("plots/Weekly-danceability.svg", format="svg")	
	

# Plot for tempo, but smaller
plt.figure(figsize=(22,4))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['tempMean','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('Tempo')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
plt.savefig("plots/Weekly-tempo.svg", format="svg")	
	

# Plot for beath strength, but smaller
plt.figure(figsize=(22,4))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['bsMean','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('Beath strength')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
	
plt.savefig("plots/Weekly-beatstrength.svg", format="svg")	
	

# Plot for loudness, but smaller
plt.figure(figsize=(22,4))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['loudMean','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('Loudness')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
plt.savefig("plots/Weekly-loudness.svg", format="svg")	





# Plot for mechanism, but smaller
plt.figure(figsize=(22,4))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['mechMean','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('Mechanism')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
plt.savefig("plots/Weekly-mechanism.svg", format="svg")		
	
	
# Plot for organism, but smaller
plt.figure(figsize=(22,4))
allPlot = sns.lineplot(x='hour', y='value', hue='variable', 
					   data=pd.melt(scaledDataFrame[['orgMean','hour']], ['hour']),
													   label=None)
#allPlot.legend_.remove()
allPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
allPlot.set(xlim=(0,167))
allPlot.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162])
allPlot.set_xticklabels(np.tile([0,6,12,18], 7))
allPlot.set_title('Organism')

colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for n in range(0,168):
	allPlot.axvspan(n, n+1, facecolor=colors[int(skm_5_labels[n])], alpha=.2)
plt.savefig("plots/Weekly-organism.svg", format="svg")
#%% Vertical bar plot of difference
	
	
diffData = pd.read_csv('data/subdivisionAudioFeatures.csv')


# Divide them due to scale differences

diffData = diffData[['subdivision', 'loudMean', 'tempMean', 'dynMean']]

morningAudio = diffData[diffData.subdivision == 'Morning']
flipped=morningAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={0:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,2))
mPlot = sns.barplot(data=morningAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.31, .31))
#plt.savefig('plots/morningDiffBig.svg', format='svg')



afternoonAudio = diffData[diffData.subdivision == 'Afternoon']
flipped=afternoonAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={1:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,2))
mPlot = sns.barplot(data=afternoonAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.31, .31))
plt.savefig('plots/afternoonDiffBig.svg', format='svg')


eveningAudio = diffData[diffData.subdivision == 'Evening']
flipped=eveningAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={2:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,2))
mPlot = sns.barplot(data=eveningAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.31, .31))
#plt.savefig('plots/eveningDiffBig.svg', format='svg')


nightAudio = diffData[diffData.subdivision == 'Night']
flipped=nightAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={3:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,2))
mPlot = sns.barplot(data=nightAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.31, .31))
#plt.savefig('plots/nightDiffBig.svg', format='svg')


emlnAudio = diffData[diffData.subdivision == 'Early Morning/Late Night']
flipped=emlnAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={4:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,2))
mPlot = sns.barplot(data=emlnAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.31, .31))
#plt.savefig('plots/emlnDiffBig.svg', format='svg')

# now for the rest

	
diffData = pd.read_csv('data/subdivisionAudioFeatures.csv')

# might have to divide them up due to the scale differences.

diffData = diffData[['subdivision', 'bsMean', 'danceMean', 'mechMean', 'orgMean', 'acousMean', 'bouncMean', 'enerMean', 'flatMean', 'instMean', 'liveMean', 'speechMean', 'valMean']]



morningAudio = diffData[diffData.subdivision == 'Morning']
flipped=morningAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={0:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,8))
mPlot = sns.barplot(data=morningAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.02, .02))
#plt.savefig('plots/morningDiffSmall.svg', format='svg')



afternoonAudio = diffData[diffData.subdivision == 'Afternoon']
flipped=afternoonAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={1:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,8))
mPlot = sns.barplot(data=afternoonAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.02, .02))
#plt.savefig('plots/afternoonDiffSmall.svg', format='svg')


eveningAudio = diffData[diffData.subdivision == 'Evening']
flipped=eveningAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={2:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,8))
mPlot = sns.barplot(data=eveningAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.02, .02))
#plt.savefig('plots/eveningDiffSmall.svg', format='svg')


nightAudio = diffData[diffData.subdivision == 'Night']
flipped=nightAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={3:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,8))
mPlot = sns.barplot(data=nightAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.02, .02))
#plt.savefig('plots/nightDiffSmall.svg', format='svg')


emlnAudio = diffData[diffData.subdivision == 'Early Morning/Late Night']
flipped=emlnAudio.transpose()
flipped = flipped.iloc[1:]
flipped = flipped.rename(columns={4:'value'})
hues = np.where(flipped.value >= 0, 'red', 'blue')
sns.set(style='whitegrid')
plt.figure(figsize=(4,8))
mPlot = sns.barplot(data=emlnAudio, orient='h', palette=hues)
mPlot.set(xlim=(-.02, .02))
#plt.savefig('plots/emlnDiffSmall.svg', format='svg')






























	
	