# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:19:36 2020

@author: olehe

This script downloads tracks from the playlists submtited in study 3.
"""

#%% Imports

import pandas as pd
import numpy as np


import sys
import spotipy
import spotipy.util as util

import time
import random
import os.path
import os
import requests


#%% Initiate token
# REPLACE BEFORE PUBLIC!!!
# This initiates a token, for my user and for my app
token=util.prompt_for_user_token('oleadrian', scope=None,
						   client_id='a91ed9edffbc4349b0ae66ad2680900a',
						   client_secret='c2f72683b7df48bc984d0f299ceb8c8c',
						   redirect_uri='http://example.com/callback/')
sp = spotipy.Spotify(auth=token)


#%% Read data


data = pd.read_csv('data/cleanStudy3data.csv')
data.drop(['Unnamed: 0','X_id'], axis=1, inplace=True)

# create a column with category, where if userResponse is 0, the category is userPrefered, otherwise is classified.

def category(row):
	if row['userResponse']==0:
		cat = row['userPrefered']
	else:
		cat = row['classified']
	return cat

data['cat'] = data.apply(category, axis=1)


#%% Start a loop wherein each track's metadata is collected.

for playlist in data.itertuples(name='Playlist'):
	thisPlaylist = playlist.playlistID
	thisParticipant = playlist.uID
	theseTracks = playlist.tracks
	theseTracks = theseTracks.replace('"', '')
	theseTracks = theseTracks.replace('[', '')
	theseTracks = theseTracks.replace(']', '')
	theseTracks = theseTracks.split(',')