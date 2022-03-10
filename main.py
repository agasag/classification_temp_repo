import numpy as np
import scipy.io as sio

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.applications import resnet

from sklearn.model_selection import train_test_split

import plotly.graph_objects as go
import pandas
import os
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime
import shutil

from skimage.io import imread
from skimage.transform import resize

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization

from weighted_loss import *

from CustomGenerator import *

# 0030 -> 0030-1 -> left/right

def fcn_TVT_folders_partition(input_path, train, val, test):
    train_dataset = []
    val_dataset = []
    test_dataset = []

    patient_folders = [f for f in os.listdir(input_path)]
    len_pat = len(patient_folders)
    train_no = np.round(train*len_pat)

    val_no = np.round(val*len_pat)
    test_no = len_pat-(train_no+val_no)

    TVTrandom = random.sample(range(len_pat), len_pat)

    train_ind = TVTrandom[0:int(train_no)]
    val_ind = TVTrandom[int(train_no)+1:(int(train_no)+1)+int(val_no)]
    test_ind = TVTrandom[(int(train_no) + 1) + int(val_no)+1:(int(train_no) + 1) + int(val_no)+1+int(test_no)]

    for i in train_ind:
        train_dataset.append(patient_folders[i])

    for i in val_ind:
        val_dataset.append(patient_folders[i])

    for i in test_ind:
        test_dataset.append(patient_folders[i])

    return train_dataset, val_dataset, test_dataset


def load_frames_single_patient(path_frames, mode):
    filenames = []  # images for CNN
    labels = []
    for paths in path_frames:
        vid_folders = [f for f in os.listdir(os.path.join('./data/frames/', paths))]
        for vid_folder in vid_folders:
            camera_folders = [f for f in os.listdir(os.path.join('./data/frames/', paths, vid_folder))]
            # left or right
            for camera in camera_folders:
                image_files = [f for f in os.listdir(os.path.join('./data/frames/', paths, vid_folder, camera))]
                for imgs in image_files:
                    filenames.append(os.path.join('./data/frames/', paths, vid_folder, camera, imgs))
                    csv_label = np.array(
                        pandas.read_csv(
                            os.path.join('./data/frames/', paths, vid_folder, camera, mode, imgs).replace('frames', 'labels').replace('png', 'csv'))).astype(
                        int)
                    labels.append(csv_label)
    return filenames, labels


def prepare_csv(path_frames, mode):
    for paths in path_frames:
        filenames = []  # images for CNN
        labels = []
        vid_folders = [f for f in os.listdir(os.path.join('./data/frames/', paths))]
        for vid_folder in vid_folders:
            print(os.path.join('./data/frames/', paths, vid_folder))
            camera_folders = [f for f in os.listdir(os.path.join('./data/frames/', paths, vid_folder))]
            # left or right
            for camera in camera_folders:
                image_files = [f for f in os.listdir(os.path.join('./data/frames/', paths, vid_folder, camera))]
                for imgs in image_files:
                    filenames.append(os.path.join('./data/frames/', paths, vid_folder, camera, imgs))
                    csv_label = np.array(
                        pandas.read_csv(
                            os.path.join('./data/frames/', paths, vid_folder, camera, mode, imgs).replace('frames', 'labels').replace('png', 'csv'))).astype(
                        int)
                    labels.append(csv_label)

                    if not os.path.exists(os.path.join('./data/npys/frames/', paths, vid_folder, camera, mode)):
                        os.makedirs(os.path.join('./data/npys/frames/', paths, vid_folder, camera, mode))
                    if not os.path.exists(os.path.join('./data/npys/labels/', paths, vid_folder, camera, mode)):
                        os.makedirs(os.path.join('./data/npys/labels/', paths, vid_folder, camera, mode))

                    np.save(os.path.join('./data/npys/labels/', paths, vid_folder, camera, mode, vid_folder), labels)
                    np.save(os.path.join('./data/npys/frames/', paths, vid_folder, camera, mode, vid_folder), filenames)


def load_frames_all_patients(path_frames):
    patient_folders = [f for f in os.listdir(path_frames)]
    all_frames = []  # images for CNN
    teeth_labels_left = None
    teeth_labels_right = None
    tongue_labels_left = None
    tongue_labels_right = None

    # patients
    for patient in patient_folders:
        vid_folders = [f for f in os.listdir(os.path.join(path_frames, patient))]
        # videos
        for vid_folder in vid_folders:
            camera_folders = [f for f in os.listdir(os.path.join(path_frames, patient, vid_folder))]
            # left or right
            for camera in camera_folders:
                image_files = [f for f in os.listdir(os.path.join(path_frames, patient, vid_folder, camera))]
                for imgs in image_files:
                    img_frames = cv2.imread(os.path.join(path_frames, patient, vid_folder, camera, imgs))
                    img_frames = cv2.resize(img_frames, (320, 240))
                    all_frames.append(img_frames)
                    print(os.path.join(path_frames, patient, vid_folder, camera, imgs))

def prepare_TVT_files(dataset, mode):
    filenames = None  # images for CNN
    labels = None

    for patient in dataset:
        vid_folders = [f for f in os.listdir(os.path.join('./data/npys/labels', patient))]
        for vid_folder in vid_folders:
            print(os.path.join('./data/npys/labels', patient, vid_folder))
            camera_folders = [f for f in os.listdir(os.path.join('./data/npys/labels', patient, vid_folder))]
            # left or right
            for camera in camera_folders:
                mode_files = [f for f in os.listdir(os.path.join('./data/npys/labels', patient, vid_folder, camera))]
                for mode in mode_files:
                    lbl_files = [f for f in os.listdir(os.path.join('./data/npys/labels', patient, vid_folder, camera, mode))]
                    for lbl in lbl_files:
                        if labels is not None:
                            labels = np.concatenate((labels, (np.load(
                                os.path.join('./data/npys/labels', patient, vid_folder, camera, mode, lbl)))), axis=0)
                            filenames = np.concatenate((filenames, np.load(
                                os.path.join('./data/npys/labels', patient, vid_folder, camera, mode, lbl).replace(
                                    'labels', 'frames'))), axis=0)
                        else:
                            labels = np.load(os.path.join('./data/npys/labels', patient, vid_folder, camera, mode, lbl))
                            filenames = np.load(os.path.join('./data/npys/labels', patient, vid_folder, camera, mode, lbl).
                                replace('labels', 'frames'))

    return labels, filenames

# binary classification or multiclass"???
train_dataset, val_dataset, test_dataset = fcn_TVT_folders_partition('./data/frames/', 0.8, 0.1, 0.1)
#prepare_csv(train_dataset, 'teeth')

labels_train, filenames_train = prepare_TVT_files(train_dataset, 'teeth')
labels_val, filenames_val = prepare_TVT_files(val_dataset, 'teeth')
labels_test, filenames_test = prepare_TVT_files(test_dataset, 'teeth')

timenow = datetime.now().strftime("%H-%M")

print('##############################################################')
print('Labels distribution (train dataset)')
print('Length: '+str(len(labels_train)))
print('0: '+str(len(labels_train)-np.sum(labels_test)))
print('1: '+str(np.sum(labels_train)))
print('##############################################################')

label_0_weight = len(labels_train)/(len(labels_train)-np.sum(labels_test))
label_1_weight = len(labels_train)/(len(labels_train)-np.sum(labels_test))

class_weights = [label_0_weight,
                 label_1_weight]


np.save(os.path.join('./data/test/', 'labels-train-'+str(timenow)), labels_train)
np.save(os.path.join('./data/test/', 'filenames-train-'+str(timenow)), filenames_train)
np.save(os.path.join('./data/test/', 'labels-val-'+str(timenow)), labels_val)
np.save(os.path.join('./data/test/', 'filenames-val-'+str(timenow)), filenames_val)
np.save(os.path.join('./data/test/', 'labels-test-'+str(timenow)), labels_test)
np.save(os.path.join('./data/test/', 'filenames-test-'+str(timenow)), filenames_test)


batch_size = 128
training_batch_generator = batch_generator(filenames_train, labels_train, batch_size)
validation_batch_generator = batch_generator(filenames_val, labels_val, batch_size)
test_batch_generator = batch_generator(filenames_test, labels_test, batch_size)

"""base_model = tf.keras.applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(240, 320, 3))
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit_generator(generator=training_batch_generator,
                    steps_per_epoch=int(np.round(len(filenames_train)/batch_size)),
                    epochs=100,
                    verbose=1,
                    validation_data=validation_batch_generator,
                    validation_steps=int(np.round(len(filenames_val)/batch_size)),
                    class_weight=class_weights)"""
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu',input_shape=(240,320,3)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu")) #Fully connected layer
model.add(BatchNormalization())
model.add(Dense(2, activation = "softmax")) #Classification layer or output layer

model.compile(optimizer="adam", loss=weighted_categorical_crossentropy(class_weights), metrics=['accuracy'])

model.summary()


model.fit_generator(generator=training_batch_generator,
                    steps_per_epoch=int(np.round(len(filenames_train)/batch_size)),
                    epochs=100,
                    verbose=1,
                    validation_data=validation_batch_generator,
                    validation_steps=int(np.round(len(filenames_val)/batch_size)))
#batch_size = 64
#my_training_batch_generator = batch_generator(X_train_filenames, y_train, batch_size)
#my_validation_batch_generator = batch_generator(X_val_filenames, y_val, batch_size)

"""filenames_train, labels_train = load_frames_single_patient(train_dataset, 'teeth')


NUM_CLASSES = 2
CHANNELS = 1

RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 3

STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

BATCH_SIZE_TRAINING = 64
BATCH_SIZE_VALIDATION = 64
BATCH_SIZE_TESTING = 1

path_frames = './data/frames/'

#datagen = ImageDataGenerator()

# load and iterate training dataset
#train_it = datagen.flow_from_directory('data/frames/', class_mode='binary', batch_size=64)
# load and iterate validation dataset
#val_it = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
# load and iterate test dataset
#test_it = datagen.flow_from_directory('data/test/', class_mode='binary', batch_size=64)

#batchX, batchy = train_it.next()
#print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))"""

