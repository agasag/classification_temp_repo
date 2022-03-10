import cv2
import numpy as np
import os

# get folders of patients
import pandas

vid_folders = [f for f in os.listdir('./data/vid/')]

all_frames = [] # images for CNN
teeth_labels_left = None
teeth_labels_right = None
tongue_labels_left = None
tongue_labels_right = None

for patient in vid_folders:
    # get vid files in patients folder
    vid_files = [f for f in os.listdir(os.path.join('./data/vid/', patient))]

    # process single video file
    for vidf in vid_files:
        print('Patient no: '+str(vidf))

        fname = os.path.join('./data/csv/', patient, vidf)

        vidcap = cv2.VideoCapture(os.path.join('./data/vid/', patient, vidf))

        if not os.path.exists(os.path.join('./data/frames/', patient, vidf[:-4], 'left')):
            os.makedirs(os.path.join('./data/frames/', patient, vidf[:-4], 'left'))

        if not os.path.exists(os.path.join('./data/frames/', patient, vidf[:-4], 'right')):
            os.makedirs(os.path.join('./data/frames/', patient, vidf[:-4], 'right'))

        if not os.path.exists(os.path.join('./data/labels/', patient, vidf[:-4], 'left', 'teeth')):
            os.makedirs(os.path.join('./data/labels/', patient, vidf[:-4], 'left', 'teeth'))

        if not os.path.exists(os.path.join('./data/labels/', patient, vidf[:-4], 'right', 'teeth')):
            os.makedirs(os.path.join('./data/labels/', patient, vidf[:-4], 'right', 'teeth'))

        if not os.path.exists(os.path.join('./data/labels/', patient, vidf[:-4], 'left', 'tongue')):
            os.makedirs(os.path.join('./data/labels/', patient, vidf[:-4], 'left', 'tongue'))

        if not os.path.exists(os.path.join('./data/labels/', patient, vidf[:-4], 'right', 'tongue')):
            os.makedirs(os.path.join('./data/labels/', patient, vidf[:-4], 'right', 'tongue'))

        csv_file_teeth_left = np.array(pandas.read_csv(os.path.join('./data/csv/teeth/left/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)
        csv_file_teeth_right = np.array(pandas.read_csv(os.path.join('./data/csv/teeth/right/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)
        csv_file_tongue_left = np.array(pandas.read_csv(os.path.join('./data/csv/tongue/left/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)
        csv_file_tongue_right = np.array(pandas.read_csv(os.path.join('./data/csv/tongue/right/', patient, vidf).replace('avi', 'csv').replace('mp4', 'csv'))).astype(int)

        for idx in range(0, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))-1):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, singleframe = vidcap.read()
            frame_name = vidf[:-4] + '-' + str(idx) + '.png'
            graysingleframe = cv2.cvtColor(singleframe, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join('./data/frames/', patient, vidf[:-4], 'left', frame_name),
                        graysingleframe[:, 0:639])
            cv2.imwrite(os.path.join('./data/frames/', patient, vidf[:-4], 'right', frame_name),
                        graysingleframe[:, 640:1280])
            #all_frames.append(singleframe[:,:,0])
            left_teeth = pandas.DataFrame(csv_file_teeth_left[idx])
            right_teeth = pandas.DataFrame(csv_file_teeth_right[idx])
            left_tongue = pandas.DataFrame(csv_file_tongue_left[idx])
            right_tongue = pandas.DataFrame(csv_file_tongue_right[idx])

            leftteethlabel_name = vidf[:-4] + '-' + str(idx) + '.csv'
            rightteethlabel_name = vidf[:-4] + '-' + str(idx) + '.csv'
            left_teeth.to_csv(os.path.join('./data/labels/', patient, vidf[:-4], 'left', 'teeth', leftteethlabel_name),
                              index=False)
            right_teeth.to_csv(os.path.join('./data/labels/', patient, vidf[:-4], 'right', 'teeth', rightteethlabel_name),
                              index=False)

            lefttonguelabel_name = vidf[:-4] + '-' + str(idx) + 'left_tongue.csv'
            righttonguelabel_name = vidf[:-4] + '-' +  str(idx) + 'right_tongue.csv'
            left_tongue.to_csv(os.path.join('./data/labels/', patient, vidf[:-4], 'left', 'tongue', leftteethlabel_name),
                              index=False)
            right_tongue.to_csv(os.path.join('./data/labels/', patient, vidf[:-4], 'right', 'tongue', righttonguelabel_name),
                              index=False)

        print(['Speaker no ' + patient + ': Teeth L/R (' + str(len(csv_file_teeth_left)) +'/' + str(len(csv_file_teeth_right)) + '), Tongue L/R: ('+ str(len(csv_file_tongue_left))+'/'+ str(len(csv_file_tongue_right))+'), vid frames: '+str(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)-1))])

        """if teeth_labels_left is not None:
            teeth_labels_left = np.concatenate((teeth_labels_left, csv_file_teeth_left))
            teeth_labels_right = np.concatenate((teeth_labels_right, csv_file_teeth_right))
            tongue_labels_left = np.concatenate((tongue_labels_left, csv_file_tongue_left))
            tongue_labels_right = np.concatenate((tongue_labels_right, csv_file_tongue_right))
        else:
            teeth_labels_left = csv_file_teeth_left
            teeth_labels_right = csv_file_teeth_right
            tongue_labels_left = csv_file_tongue_left
            tongue_labels_right = csv_file_tongue_right"""





