# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:33:13 2019

@author: Ole Adrian Heggli

This script pre-processes the MSSD

"""

#%% Imports
import pandas as pd
import os
import datetime
import time

#%% Definitions
# Function for getting weekday out of date

def getWeekday(date):
    try:
        this_day=datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%a')
    except:
        print('Error in determining weekday, error in format. Setting weekday to NA')
        this_day='NA'
    return this_day

# Function for adding unique hour of the week
def addUniqueHour(row):
    if row['weekday']=='Mon':
        return row['hour_of_day']
    elif row['weekday']=='Tue':
        return row['hour_of_day']+24
    elif row['weekday']=='Wed':
        return row['hour_of_day']+48
    elif row['weekday']=='Thu':
        return row['hour_of_day']+72
    elif row['weekday']=='Fri':
        return row['hour_of_day']+96
    elif row['weekday']=='Sat':
        return row['hour_of_day']+120
    elif row['weekday']=='Sun':
        return row['hour_of_day']+144
    elif row['weekday']=='NA':
        return -1

#%% Initializing variables and getting files

# Name for the output files
output_name='_collatedDataWithInfoAndUnique_allFeatures.csv'
count_files=1
processed_files=1

# Path of all the log files.
# Replace with your own path
datapathTraining='d:\\SpotifyData\\data\\training_set\\'

# Path for output data.
savepath='e:\\SpotifyData\\'

files=[]



# Read the track features.
featurespath='d:\\SpotifyData\\data\\track_features\\'
set_1=pd.read_csv(featurespath+'tf_000000000000.csv', header=0)
set_2=pd.read_csv(featurespath+'tf_000000000001.csv', header=0)

# Concatanate into one
feature_data=pd.concat([set_1, set_2], sort=False)
# Clear variables.
set_1=None
set_2=None



# Get a list of all the files to use
# r is root, d is directory, f is file
for r, d, f in os.walk(datapathTraining):
    for file in f:
        files.append(os.path.join(r, file))      
# Just check that this hasn't been run twice by accident

assert len(files) == 660, 'Wrong number of input files'

# Keep count of number of tracks
number_of_files=len(files) 
number_of_tracks_heard=0



 
    
#%% Main processing loop

# Note that this could be parallelized
# At the moment, it is set up to just run in the background (for a long time)
    
for file in files:
    
    start_marker=time.time()
    print('Now processing file', file)
    print('There are', number_of_files, 'files left') 
    number_of_files -= 1
    
    # Read only columns we want
    cols_of_interest = [0,3,6,7,12,13,15,16]     
    this_data = pd.read_csv(file, header=0, usecols=cols_of_interest)
    
    # Filter to get only complete listens and max amount of seeks
    filtered_data = this_data.query('skip_3 == True or not_skipped == True and hist_user_behavior_n_seekfwd < 2 and hist_user_behavior_n_seekback < 2')
    
    # Clear non-filtered data
    this_data = None
    
    # Filter data for size considerations
    filtered_data.drop(columns=['skip_3', 'not_skipped', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback'], axis=1, inplace=True)
    
    # Update tracking variables
    number_of_tracks_heard += filtered_data.shape[0]
    
    # Add weekday
    filtered_data['weekday']=filtered_data.apply(lambda row: getWeekday(row['date']), axis=1)
    filtered_data['unique']=filtered_data.apply(lambda row: addUniqueHour(row), axis=1)
    
    # Do a lookup on the musical features we're interested in
    filtered_data=pd.merge(filtered_data, 
                           feature_data[['track_id', 'beat_strength', 'loudness', 'mechanism', 'organism', 'tempo', 'acousticness', 'bounciness', 'danceability', 'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'liveness', 'speechiness', 'valence']], 
                           how='left', 
                           left_on='track_id_clean',
                           right_on='track_id')
    

    # Save the resulting datafile
    if os.path.isfile(savepath+str(count_files)+output_name):
        # If the file exist, append data
        with open(savepath+str(count_files)+output_name, 'a') as f:
            filtered_data.to_csv(f, index=None, header=False)
        # check size of csv and make new if it gets too big
        size=os.path.getsize(savepath+str(count_files)+output_name)

        if (size/1e9)>20:
            f.close()
            count_files += 1
            print('Filled an output file')
        
    else:
        # If the file doesn't exist, create it.
        filtered_data.to_csv(savepath+str(count_files)+output_name, index=None, header=True)
    
    # Clear the data
    filtered_data=None
    
    # Report time
    end_marker=time.time()
    print('This raw data file took', round(end_marker-start_marker), 'seconds to process.')
    # Estimate processing time (very roughly)
    print('Estimated processing time is',
          (((number_of_files+1)*
          (round(end_marker-start_marker)))/
          3600), 'hours.')



