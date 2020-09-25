# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:09:59 2020

@author: olehe

This script calculates the modal onset of the clusters.

"""


#%% Imports

import numpy as np
import pandas as pd
from scipy import stats


#%% Calculate modal onsets


# Read in the k-means labels
labels=pd.read_pickle('data/kmeansLabelsForKequal5')
kmeansInd = labels.values

hourRep = np.tile(np.arange(0,24),7)

morningArray = []
afternoonArray = []
nightArray = []
earlymorningArray = []
eveningArray = []

thisHolder = 5
thisPrev = 5


# Hardcoded for-loop to sort them.
for i in range(len(hourRep)):
	thisHolder = kmeansInd[i]
	if thisHolder != thisPrev:
		if thisHolder == 4:
			morningArray.append(hourRep[i])
		elif thisHolder == 1:
			afternoonArray.append(hourRep[i])
		elif thisHolder == 2:
			nightArray.append(hourRep[i])
		elif thisHolder == 0:
			earlymorningArray.append(hourRep[i])
		elif thisHolder == 3:
			eveningArray.append(hourRep[i])
	thisPrev = thisHolder

# NOTE! As the week is circular, we have one change to 2 right before the last datapoint.
# Manually fix this by removing the first one.
nightArray.pop(0)




stats.mode(earlymorningArray)
stats.circstd(earlymorningArray, 23, 0)
stats.mode(morningArray)
stats.circstd(morningArray, 23, 0)
stats.mode(afternoonArray)
stats.circstd(afternoonArray, 23, 0)
stats.mode(eveningArray)
stats.circstd(eveningArray, 23, 0)
stats.mode(nightArray)
stats.circstd(nightArray, 23, 0)


