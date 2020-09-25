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

Some package versions:
	Python 3.7.6
	cudaatoolkit 10.0.130
	tensorflow 2.0.0
	tensorflow-gpu 2.0.0


This project is not published or preprinted yet.
If you use it, please cite the github repository until a paper has been published.


"""

#%% Imports
# Many of these imports are not needed in this version of the script.
# Remove unused imports on line 48:64, and 68:76 if needed.

import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, LeakyReLU, Lambda, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, Adadelta, Nadam, SGD, Adamax, Adadelta
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

import time

import pickle

import spotipy
import spotipy.util as util
import random

#%% Some definitions

# Simple function for remapping variables
# Copied from PenguinTD on Stackoverflow
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


model = load_model('pretrained/')


#%% Make a new model without activity regularization
# TFJS does not support activity regularization, so remake the model without them

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

#%% Check the models on the testing part of the dataset

# Read data
testingData = pd.read_pickle('testingDataset.pkl')
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



#%% Example for evaluating the model on one track
# Chose a random integer to test a track
thisTrack = testingDataValues[4000,:].reshape(1,6)

thisClassification = model_2.predict(thisTrack)

printPrediction(np.argmax(thisClassification))

