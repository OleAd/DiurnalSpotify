# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:44:09 2020

@author: olehe

This script takes the American Time Use Survey just to make a point for the paper


"""

#%% Imports

import pandas as pd
import numpy as np
from sklearn import preprocessing


import seaborn as sns
from matplotlib import pyplot as plt

from datetime import datetime, timedelta
from collections import namedtuple



#%% Load the data
# Lexicon here: https://www.bls.gov/tus/lexiconnoex0319.pdf
# Data description here: https://www.bls.gov/tus/atusintcodebk0319.pdf

timeUseData = pd.read_csv('AmericanTimeUseSurvey/atusact-0319/atusact_0319.dat', low_memory=False)
# variables we want is tucaseid, tustarttime, tustoptime, trcodep

timeUseDataSmall = timeUseData[['TUCASEID', 'TUSTARTTIM', 'TUSTOPTIME', 'TRCODEP']]

# round start down, and stop up

timeUseDataSmall['StartTime'] = pd.to_datetime(timeUseDataSmall['TUSTARTTIM'], format='%H:%M:%S')
timeUseDataSmall['StartTimeHour'] = timeUseDataSmall['StartTime'].dt.hour
timeUseDataSmall['EndTime'] = pd.to_datetime(timeUseDataSmall['TUSTOPTIME'], format='%H:%M:%S')
timeUseDataSmall['EndTimeHour'] = timeUseDataSmall['EndTime'].dt.hour


	
#%% Now the actual test

testData = timeUseDataSmall.head(1000)

# probably got to do this per individual

uniqueIndividuals = timeUseDataSmall['TUCASEID'].unique()
# activity always starts at 4? Yeah, take that into account
holder=[]
for individual in uniqueIndividuals:
	activityOverview=np.empty(24)
	activityOverview[:] = np.NaN
	thisData = timeUseDataSmall[timeUseDataSmall['TUCASEID'] == individual]
	thisLoc = 0
	for time in thisData['StartTimeHour']:
		#print(time)
		thisPosition = time-4
		thisPosition = thisPosition % 24
		activityOverview[thisPosition]=thisData.iloc[thisLoc].TRCODEP
		thisLoc += 1
	# now expand
	
	# now return a dataframe
	outputFrame = pd.DataFrame(activityOverview)
	outputFrame.fillna(method='ffill', inplace=True)
	outputFrame.columns=['activity']
	hours = (np.arange(0,24) + 4) % 24
	outputFrame['TUCASEID'] = individual
	outputFrame['hour'] = hours
	holder.append(outputFrame)
		

# now join the dataframes.	
	
results = pd.concat(holder)


results.to_csv('DailyActivitiesAmericanTimeUseSurvey.csv', encoding='utf-8')

#%% Now for analysis

# First filter out all the 5-codes, as these are lacking data.
# first, reset index

results.reset_index(inplace=True)

missingData = results[results['activity'] >= 500101]

filteredData = results.drop(missingData.index)
sleep1Data = filteredData[filteredData['activity'] == 10101]
filteredData.drop(sleep1Data.index, inplace=True)
sleep2Data = filteredData[filteredData['activity'] == 10199]
filteredData.drop(sleep2Data.index, inplace=True)

sleep3Data = filteredData[filteredData['activity'] == 10102]
filteredData.drop(sleep3Data.index, inplace=True)


filteredData.to_csv('DailyActivitiesAmericanTimeUseSurvey_filtered.csv', encoding='utf-8')


# Now, count unique per hour or something.
filteredData['activityCat'] = filteredData['activity'].astype('category')

# consider rounding the activities 
bins = pd.IntervalIndex.from_tuples([(10000, 19999), (20000,29999), (30000, 39999), (40000, 49999), (50000, 59999), (60000,69999), (70000,79999), (80000,89999), (90000,99999),(100000,109999),(110000,119999),(120000,129999),(130000,139999),(140000,149999),(150000,159999),(160000,169999)])

#binned = pd.cut(filteredData['activity'], bins, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], include_lowest=True)

binned = pd.cut(filteredData['activity'], bins, labels=False, include_lowest=True)

filteredData['activityInterval'] = binned



# this gives the absolute number of unique activities per hour
activityUniqueCount = filteredData.groupby('hour')['activity'].nunique()

# however, we want some sort of "variability" measure
# in this case, we could look at the proportions of each unique activity
# and then calculate something like the distribution-ish of each.

activityProportionCount = filteredData[['activity', 'hour']].groupby(['activity', 'hour']).agg('count')

# fuck it, do a for loop

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


# now find a way to calculate a sort of diversity value
	
# inspect data a bit
	
inspect = hourActivityHolder[0].head(10)
sns.lineplot(np.arange(1,10), inspect['proportion'])


# get data together

hoursTogether = pd.concat(hourActivityHolder)


plt.figure(figsize=(8,8))
stdsHourPlot = sns.lineplot(x='activityIdent', y='proportion', hue='hour',
							data=hoursTogether[hoursTogether['activityIdent']<5],
							label=None)
	
# how about, top 20 activities in terms of proportions, then a linear fit
# diversity would be how "flat" the line is
# no, instead let's go for a filtering-option, where we get rid of the first or couple of first activities
	


diversityList = []

slope=[]
intercept=[]

testingList = []
botIQRList = []
topIQRList = []
IQRList = []

# THIS IS WHAT WE GO FOR
for hour in hourActivityHolder:
	thisData = hour.head(50)
	#thisData = hour
	thisValues = thisData['proportion']
	thisValues.reset_index(inplace=True, drop=True)
	#thisValues = thisValues[1:]
	a,b = np.polyfit(thisValues.index, thisValues, 1)
	thisStd = np.std(thisValues)
	diversityList.append(thisStd)
	slope.append(a)
	intercept.append(b)
	diffFall = np.median(np.diff(thisValues))
	testingList.append(diffFall)
	botIQR, topIQR = np.percentile(np.diff(thisValues), [40, 60])
	IQR = topIQR-botIQR
	IQRList.append(IQR)
	botIQRList.append(botIQR)
	topIQRList.append(topIQR)
	
'''
sns.lineplot(np.arange(0,24), diversityList)

sns.lineplot(np.arange(0,24), intercept)
sns.lineplot(np.arange(0,24), slope)
'''

# this is the one to use

# Also make a smoothed line, i.e. 3 hours

from scipy.interpolate import make_interp_spline, BSpline

interpActSpline = make_interp_spline(np.arange(0,24), testingList, k=3)
interpAct = interpActSpline(np.arange(0,24))
# nope, this just interpolates

toSmooth = np.tile(testingList,3)
toSmooth = pd.DataFrame(toSmooth)
toSmooth = toSmooth.rolling(4, center=True).mean()

smoothed = toSmooth.iloc[24:48]


actData = pd.DataFrame(testingList)
actData.rename(columns={0:'Activity'}, inplace=True)
actData['ActivitySmooth'] = smoothed.values
actData['hour'] = np.arange(0,24)

# first calculate IQR
plt.figure(figsize=(8,8))
diversityPlot = sns.lineplot(x='hour', y='value', hue='variable',
							 data=pd.melt(actData, ['hour']))
#diversityPlot.fill_between(np.arange(0,24), botIQRList, topIQRList, alpha=0.3)
diversityPlot.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
diversityPlot.set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
diversityPlot.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21])
diversityPlot.set_title('24-hour activity variability')
plt.savefig("plots/diversityPlot.svg", format="svg")

# save the testingList as diversitylist
diversityFrame = pd.DataFrame(testingList)
diversityFrame.rename(columns={0:'Diversity'}, inplace=True)
diversityFrame.to_pickle('diversityActivityValue.pkl')


























