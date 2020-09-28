# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:44:09 2020

@author: olehe

This script analyses activities from the American Time Use Survey (ATUS)

"""

#%% Imports

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt




#%% Load the data
# Lexicon here: https://www.bls.gov/tus/lexiconnoex0319.pdf
# Data description here: https://www.bls.gov/tus/atusintcodebk0319.pdf

# To run this script, aquire the atusact_0319 dataset first.
timeUseData = pd.read_csv('AmericanTimeUseSurvey/atusact-0319/atusact_0319.dat', low_memory=False)
# variables we want is tucaseid, tustarttime, tustoptime, trcodep
# These describe the individual ID, activity start and stop, and the code of the activity.

timeUseDataSmall = timeUseData[['TUCASEID', 'TUSTARTTIM', 'TUSTOPTIME', 'TRCODEP']]

# Now adapt data to round to hours.
timeUseDataSmall['StartTime'] = pd.to_datetime(timeUseDataSmall['TUSTARTTIM'], format='%H:%M:%S')
timeUseDataSmall['StartTimeHour'] = timeUseDataSmall['StartTime'].dt.hour
timeUseDataSmall['EndTime'] = pd.to_datetime(timeUseDataSmall['TUSTOPTIME'], format='%H:%M:%S')
timeUseDataSmall['EndTimeHour'] = timeUseDataSmall['EndTime'].dt.hour


	
#%% Extend activity to per hour

# Get all unique individuals
uniqueIndividuals = timeUseDataSmall['TUCASEID'].unique()
# Reported activity always starts at 04:00
holder=[]
for individual in uniqueIndividuals:
	# Initiate NaN 24-hour array.
	activityOverview=np.empty(24)
	activityOverview[:] = np.NaN
	thisData = timeUseDataSmall[timeUseDataSmall['TUCASEID'] == individual]
	thisLoc = 0
	for time in thisData['StartTimeHour']:
		# Minus 4 due to reporting always starting at 04:00.
		thisPosition = time-4
		thisPosition = thisPosition % 24
		activityOverview[thisPosition]=thisData.iloc[thisLoc].TRCODEP
		thisLoc += 1
	
	# Return as a dataframe
	outputFrame = pd.DataFrame(activityOverview)
	# Forward fill the hours.
	outputFrame.fillna(method='ffill', inplace=True)
	outputFrame.columns=['activity']
	hours = (np.arange(0,24) + 4) % 24
	outputFrame['TUCASEID'] = individual
	outputFrame['hour'] = hours
	holder.append(outputFrame)
		

# Concatenate dataframe
results = pd.concat(holder)
# Save data
results.to_csv('DailyActivitiesAmericanTimeUseSurvey.csv', encoding='utf-8')

#%% Analysis

# First filter out all the 5-codes, as these are missing/bad data code
# first, reset index
results.reset_index(inplace=True)

missingData = results[results['activity'] >= 500101]
filteredData = results.drop(missingData.index)

# Then filter out all sleep-related
sleep1Data = filteredData[filteredData['activity'] == 10101]
filteredData.drop(sleep1Data.index, inplace=True)
sleep2Data = filteredData[filteredData['activity'] == 10199]
filteredData.drop(sleep2Data.index, inplace=True)
sleep3Data = filteredData[filteredData['activity'] == 10102]
filteredData.drop(sleep3Data.index, inplace=True)

# Save data
filteredData.to_csv('DailyActivitiesAmericanTimeUseSurvey_filtered.csv', encoding='utf-8')


# Calculate proportions per hour
hours = np.arange(0,24)
hourActivityHolder=[]

for hour in hours:
	thisData = filteredData[filteredData['hour'] == hour]
	proportions = thisData['activity'].value_counts(normalize=True)
	output = pd.DataFrame(proportions)
	output.rename(columns={'activity':'proportion'}, inplace=True)
	output['activity'] = output.index
	output['hour'] = hour
	activityIdent = np.arange(0,len(proportions))
	output['activityIdent'] = activityIdent
	hourActivityHolder.append(output)



# Inspect data	
inspect = hourActivityHolder[0].head(10)
sns.lineplot(np.arange(1,10), inspect['proportion'])


# get data together

hoursTogether = pd.concat(hourActivityHolder)

# Calculate the activity diversity index

adiList = []

for hour in hourActivityHolder:
	# Choose top 50 activities
	thisData = hour.head(50)
	thisValues = thisData['proportion']
	thisValues.reset_index(inplace=True, drop=True)
	diffFall = np.median(np.diff(thisValues))
	adiList.append(diffFall)
	

# Now also make a smoothed one
# Tile three times, so we average circularly
toSmooth = np.tile(adiList,3)
toSmooth = pd.DataFrame(toSmooth)
toSmooth = toSmooth.rolling(4, center=True).mean()
# Choose middle
smoothed = toSmooth.iloc[24:48]

# Put into dataframe
actData = pd.DataFrame(adiList)
actData.rename(columns={0:'Activity'}, inplace=True)
actData['ActivitySmooth'] = smoothed.values
actData['hour'] = np.arange(0,24)


#%% Plot and save data
plt.figure(figsize=(8,8))
diversityPlot = sns.lineplot(x='hour', y='value', hue='variable',
							 data=pd.melt(actData, ['hour']))
diversityPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
diversityPlot.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
diversityPlot.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21])
diversityPlot.set_title('24-hour activity variability')
plt.savefig("plots/diversityPlot.svg", format="svg")

# save the testingList as diversitylist
diversityFrame = pd.DataFrame(adiList)
diversityFrame.rename(columns={0:'AID'}, inplace=True)
diversityFrame.to_pickle('aidValues.pkl')


























