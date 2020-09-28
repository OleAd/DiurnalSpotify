# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:57:06 2020

@author: olehe


This script trains the neural network model


"""

#%% Imports

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


import pandas as pd
import numpy as np



#%% Define remap

# function for remapping variables
def remap( x, oMin, oMax, nMin, nMax ):
	#range check
	if oMin == oMax:
		print("Warning: Zero input range")
		return None
	
	if nMin == nMax:
		print("Warning: Zero output range")
		return None
	
	#check reversed input range
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



#%% Read the data

# Read class weights
classweights = np.load('data/trainingClassweights.npy')
# Read training dataset
trainingData = pd.read_pickle('data/trainingDataset.pkl')
trainingData = shuffle(trainingData)
classweightsDict = {0:classweights[0],
					1:classweights[1],
					2:classweights[2],
					3:classweights[3],
					4:classweights[4]}


# Get only features available from the Spotify API
trainingDataValues = trainingData[['danceability', 'energy', 'loudness', 'liveness', 'valence', 'tempo']]
trainingDataValues = np.array(trainingDataValues)

# Remap
trainingDataValues[:,0] = remap(trainingDataValues[:,0], 0, 1, -1, 1)
trainingDataValues[:,1] = remap(trainingDataValues[:,1], 0, 1, -1, 1)
trainingDataValues[:,2] = remap(trainingDataValues[:,2], -60, 12, -1, 1)
trainingDataValues[:,3] = remap(trainingDataValues[:,3], 0, 1, -1, 1)
trainingDataValues[:,4] = remap(trainingDataValues[:,4], 0, 1, -1, 1)
trainingDataValues[:,5] = remap(trainingDataValues[:,5], 40, 220, -1, 1)

# Get labels
trainingDataLabels = trainingData['subdivision']
trainingDataLabels = np.array(trainingDataLabels)



# Labels to categorical
trainingDataLabels = tf.keras.utils.to_categorical(trainingDataLabels)

# Split into training and validation
x_train, x_valid, y_train, y_valid = train_test_split(trainingDataValues, trainingDataLabels, test_size=0.2, shuffle=True)


#%% Initiate the model
# Use only sigmoid activation due to TFJS
model = Sequential()
model.add(Dense(64, input_shape=(6,), activation='sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='sigmoid', activity_regularizer=l1(10e-3)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='sigmoid', activity_regularizer=l1(10e-3)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='sigmoid', activity_regularizer=l1(10e-3)))
model.add(Dense(5, activation='softmax'))

# Early stop if no improvement in validation accuracy, 30 epochs
early_stop = EarlyStopping(monitor='val_accuracy', patience=30, verbose=1)

# Start learning rate
learningRate=0.0001

# Add a model checkpoint, save best only.
mcp_saveAll = ModelCheckpoint('SML_NN_ModelCandidatex_.{epoch:02d}-{accuracy:.2f}.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
model.compile(Adam(lr=learningRate), loss='categorical_crossentropy', metrics=['accuracy'])


#%% Train the model


history = model.fit(x_train, y_train,
					batch_size=32, epochs=500,
					validation_data=(x_valid, y_valid), shuffle=True,
					class_weight=classweightsDict,
					verbose=1,
					callbacks=[mcp_saveAll, early_stop])









