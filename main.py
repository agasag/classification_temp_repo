import cv2
import numpy as np
import os
from os import listdir

# get folders of patients
import pandas

vid_folders = [f for f in listdir('./data/vid/')]

all_frames = [] # images for CNN
teeth_labels_left = []
teeth_labels_right = []
tongue_labels_left = []
tongue_labels_right = []

for patient in vid_folders:
    # get vid files in patients folder
    vid_files = [f for f in listdir(os.path.join('./data/vid/', patient))]

    # process single video file
    for vidf in vid_files:

        fname = os.path.join('./data/csv/', patient, vidf)
        csv_file_teeth_left = np.array(pandas.read_csv(os.path.join('./data/csv/teeth/left/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)
        csv_file_teeth_right = np.array(pandas.read_csv(os.path.join('./data/csv/teeth/right/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)
        csv_file_tongue_left = np.array(pandas.read_csv(os.path.join('./data/csv/tongue/left/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)
        csv_file_tongue_right = np.array(pandas.read_csv(os.path.join('./data/csv/tongue/right/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)

        vidcap = cv2.VideoCapture(os.path.join('./data/vid/', patient, vidf))
        success, singleframe = vidcap.read()
        count = 0
        while success:
            all_frames.append(singleframe)
            success, image = vidcap.read()
            count += 1

