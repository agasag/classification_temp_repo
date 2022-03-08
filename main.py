import numpy as np
import scipy.io as sio

import cupy
from cupyx.scipy.ndimage import rotate

from tensorflow.keras import losses
import tensorflow as tf

from sklearn.model_selection import train_test_split

import plotly.graph_objects as go

from os import listdir
from os.path import isfile, join

import time
import datetime


def load_interpolations(filelist):
    dataset_interp_all = []

    angles = np.random.randint(0, 359, (len(filelist), 10))

    for file, file_angles in zip(filelist, angles):
        mat_interpolation = sio.loadmat(file)
        interpolation = mat_interpolation['Tintrp_mesh']
        gpu_interpolation = cupy.array(interpolation)

        dataset_interp = []
        for angle in file_angles:
            # interpolation_rotate = ndimage.rotate(interpolation, angle, reshape=False)
            interpolation_rotate = rotate(gpu_interpolation, angle, reshape=False)

            interpolation_rotate[np.isnan(interpolation_rotate)] = 0
            interpolation_rotate = interpolation_rotate[:, :, ::-1]
            dataset_interp.append(interpolation_rotate)

        dataset_interp_all.append(cupy.ndarray.get(np.stack(dataset_interp, axis=0)))

    dataset_interp = np.concatenate(dataset_interp_all, axis=0)

    dataset_interp = preprocessing.source_hole_filling(dataset_interp, 50)

    dataset_interp = dataset_interp - 20
    dataset_interp[dataset_interp < 0] = 0

    return dataset_interp, angles


def load_reconstructions(filelist, angles):
    dataset_reconstr_all = []

    for file, file_angles in zip(filelist, angles):
        mat_reconstruction = sio.loadmat(file)
        reconstruction = mat_reconstruction['ir3d']

        dataset_reconstr = []
        gpu_reconstruction = cupy.array(reconstruction)
        for angle in file_angles:
            # reconstruction_rotate = ndimage.rotate(reconstruction, angle, reshape=False)
            reconstruction_rotate = rotate(gpu_reconstruction, angle-90, reshape=False, order=1)

            dataset_reconstr.append(reconstruction_rotate)

        dataset_reconstr_all.append(cupy.ndarray.get(np.stack(dataset_reconstr, axis=0)))

    dataset_reconstr = np.concatenate(dataset_reconstr_all, axis=0)

    dataset_reconstr = dataset_reconstr - 20
    dataset_reconstr[dataset_reconstr < 0] = 0


    return dataset_reconstr


def generator(files, batch_size):
    L = len(files)

    files_reconst = [f.replace('_model', '_reconstruction') for f in files]
    #this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            Y, angles = load_interpolations(files[batch_start:limit])
            X = load_reconstructions(files_reconst[batch_start:limit], angles)

            #print(X.shape)

            #X = np.expand_dims(np.concatenate(X, axis=2).transpose(), axis=3)
            #Y = np.expand_dims(np.concatenate(Y, axis=2).transpose(), axis=3)

            if is_conv3d == True:
                Y = np.expand_dims(Y, axis=4)
                X = np.expand_dims(X, axis=4)

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def load_single_interpolation(file):
    dataset_interp = []
    mat_interpolation = sio.loadmat(file)
    interpolation = mat_interpolation['Tintrp_mesh']
    interpolation_rotate = interpolation

    interpolation_rotate[np.isnan(interpolation_rotate)] = 0
    interpolation_rotate = preprocessing.source_hole_filling(interpolation_rotate, 50)
    #dataset_interp.append(interpolation_rotate)
    interpolation_rotate = interpolation_rotate - 20
    interpolation_rotate[interpolation_rotate<0] = 0
    #dataset_interp = np.stack(dataset_interp, axis=0)
    #dataset_interp = dataset_interp - 20
    #dataset_interp[dataset_interp < 0] = 0

    return interpolation_rotate#dataset_interp



path = './CNN_100x100x50/_model/' #'./CNN_100x100x50/_model_mat/'
path_r = './CNN_100x100x50/_reconstruction/' #'./CNN_100x100x50/_reco_mat/'
#path_r = './CNN_100x100x50/_recos_new/'

files_interpolation = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
files_reconstruction = [join(path_r, f) for f in listdir(path_r) if isfile(join(path_r, f))]

is_conv3d = False

#X_train, X_test, y_train, y_test = train_test_split(files_reconstruction,
#                                                    files_interpolation, test_size=0.1, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(files_reconstruction, files_interpolation,
                                                    test_size=0.2, random_state=1) # 80% to treningowe, 20% to testowe + walidacyjne

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                test_size=0.5, random_state=1)  # 10% to testowe, 10% to walidacyjne

date_save = datetime.datetime.now()
np.save('./data/X_train_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), X_train)
np.save('./data/X_test_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), X_test)
np.save('./data/X_val_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), X_val)

#
#
# # for conv3D
# if is_conv3d:
#     dataset_interp = np.expand_dims(dataset_interp, axis=4)
#     dataset_reconstr = np.expand_dims(dataset_reconstr, axis=4)
#

# model = TemperatureAutoencoder(interpolation.shape)
# model.compile(optimizer='adam', loss=losses.MeanSquaredError())
# model.fit(X_train, y_train, epochs=100, shuffle=True,
#           validation_data=(X_test, y_test))

#model = TemperatureAutoencoderAS((100, 100, 50, 1))
model = TemperatureAutoencoder((100, 100, 50))
model.compile(optimizer='adam', loss=losses.MeanSquaredError())
# training_history = model.fit(X_train, y_train,
#                              epochs=500,
#                              batch_size=64,
#                              shuffle=True,
#                              validation_data=(X_test, y_test))
# minibatch = 32
val_steps = np.ceil((len(y_val)/32))
##training_history = model.fit(generator(y_train, 8), steps_per_epoch=np.ceil(len(files_interpolation)/8), epochs=500)
training_history = model.fit(generator(y_train, 32), steps_per_epoch=np.ceil(len(files_interpolation)/32), epochs=500,
                             validation_data=generator(y_val, 32), validation_steps=val_steps) #len(y_val) validation_data=generator(y_val, 32),
# validation_steps= number of validation samples/batch_size


model.save('.\models\CNN_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

test_case_ind = 5
test = np.expand_dims(load_single_interpolation(X_test[test_case_ind]), axis=0)
interp_test = load_single_interpolation(y_test[test_case_ind])

encoded_img = model.encoder(test).numpy()
decoded_img = model.decoder(encoded_img).numpy()
decoded_img = decoded_img[0]

if is_conv3d:
    test_ex = interp_test[:, :, :, 0]
    output_ex = decoded_img[:, :, :, 0]
else:
    test_ex = interp_test
    output_ex = decoded_img

volume_plot.show(test_ex)
volume_plot.show(output_ex)

difference = test_ex - output_ex
volume_plot.show(difference)

fig = go.Figure()
fig.add_trace(go.Scatter(y=training_history.history["loss"], name="loss"))
#fig.add_trace(go.Scatter(y=training_history.history["val_loss"], name="val_loss"))
fig.show()
