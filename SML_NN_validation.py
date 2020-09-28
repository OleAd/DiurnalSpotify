# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:31:46 2020

@author: olehe

This script loads a pre-trained neural network for classifying Spotify tracks.
The network is trained to categorize Spotify audio features into 5 times of the day:
	0 - Late night/early morning -  03:00 to 05:00 (technically, 04:59)
	1 - Morning - 05:00 to 11:00 (technically, 10:59)
	2 - Afternoon - 11:00 to 19:00 (technically, 18:59)
	3 - Evening - 19:00 to 23:00 (technically, 22:59)
	4 - Night - 23:00 to 03:00 (technically, 02:59)

	
It takes as input:
	danceability
	energy
	loudness
	liveness
	valence
	tempo
	
These values are then scaled individually, to ensure it works on data directly
from the Spotify API.
 
The neural network is designed to be compatible with Tensorflow JS.

Some key package versions:
	Python 3.7.6
	cudaatoolkit 10.0.130
	tensorflow 2.0.0
	tensorflow-gpu 2.0.0

"""

#%% Imports

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np


#%% Defining functions


# Simple function for remapping variables
# Adapted from PenguinTD on Stackoverflow
def remap( x, oMin, oMax, nMin, nMax ):
	# range check
	if oMin == oMax:
		print("Warning: Zero input range")
		return None
	
	if nMin == nMax:
		print("Warning: Zero output range")
		return None
	
	# check reversed input range
	reverseInput = False
	oldMin = min( oMin, oMax )
	oldMax = max( oMin, oMax )
	if not oldMin == oMin:
		reverseInput = True
	
	#check reversed output range
	reverseOutput = False   
	newMin = min( nMin, nMax )
	newMax = max( nMin, nMax )
	if not newMin == nMin :
		reverseOutput = True
	
	portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
	if reverseInput:
		portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)
	
	result = portion + newMin
	if reverseOutput:
		result = newMax - portion
	
	return result

# This function just spells out the prediction
def printPrediction(value):
	if value==0:
		print('Main prediction is late night/early morning')
	elif value==1:
		print('Main prediction is morning')
	elif value==2:
		print('Main prediction is afternoon')
	elif value==3:
		print('Main prediction is evening')
	elif value==4:
		print('Main prediction is night')
	else:
		print('error')



#%% Load the model
# This loads the pre-trained model.
# Note that this model includes activity regularization

model = load_model('neural_network/')


#%% Make a new model without activity regularization
# TFJS does not support activity regularization

model_2 = Sequential()
model_2.add(Dense(64, input_shape=(6,), activation='sigmoid'))
model_2.add(Dropout(0.3))
model_2.add(Dense(128, activation='sigmoid'))
model_2.add(Dropout(0.3))
model_2.add(Dense(64, activation='sigmoid'))
model_2.add(Dropout(0.3))
model_2.add(Dense(32, activation='sigmoid'))
model_2.add(Dense(5, activation='softmax'))
model_2.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_2.set_weights(model.get_weights())

#%% Test both models on the testing (holdout) dataset

# Read data
testingData = pd.read_pickle('data/holdoutDataset.pkl')
testingDataValues = testingData[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]
testingDataValues = np.array(testingDataValues)

# Remap values
testingDataValues[:,0] = remap(testingDataValues[:,0], 0, 1, -1, 1)
testingDataValues[:,1] = remap(testingDataValues[:,1], 0, 1, -1, 1)
testingDataValues[:,2] = remap(testingDataValues[:,2], -60, 12, -1, 1)
testingDataValues[:,3] = remap(testingDataValues[:,3], 0, 1, -1, 1)
testingDataValues[:,4] = remap(testingDataValues[:,4], 0, 1, -1, 1)
testingDataValues[:,5] = remap(testingDataValues[:,5], 40, 220, -1, 1)

# Create labels
testingDataLabels = testingData['subdivision']
testingDataLabels = np.array(testingDataLabels)

# Change labels to categorical
testingDataLabels = tf.keras.utils.to_categorical(testingDataLabels)

# Evaluate models
results_1 = model.evaluate(testingDataValues, testingDataLabels, batch_size=128)
results_2 = model_2.evaluate(testingDataValues, testingDataLabels, batch_size=128)

print('Model 1 with regularization performed with accuracy of: ' + str(round(results_1[1],3)) + ' and loss of: ' + str(round(results_1[0],3)))
print('Model 2 without regularization performed with accuracy of: ' + str(round(results_2[1],3)) + ' and loss of: ' + str(round(results_2[0],3)))



#%% Example for evaluating the model on one track
# Take a single track from the testing dataset
thisTrack = testingDataValues[4000,:].reshape(1,6)

thisClassification = model_2.predict(thisTrack)

printPrediction(np.argmax(thisClassification))

