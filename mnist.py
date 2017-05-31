# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 12:06:28 2017

@author: Chad
"""

# Code to try building a DNN for the MNIST dataset using keras

import numpy as np
import csv
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.initializations import normal, identity

#load Data Here
print("Fetching Training Data...")
data = np.asarray(list(csv.reader(open("./data/train.csv","rb"),delimiter=',')))
MNIST_data = np.delete(data,0,0)

Ytrain = (np.array(MNIST_data[:,0], dtype=np.int32))
Ytrain = np.reshape(Ytrain,Ytrain.shape[0])

Xtrain = (np.array(MNIST_data[:,1:], dtype=np.float32))
Xtrain = np.reshape(Xtrain,(Xtrain.shape[0],1,28,28))

#Xtrain = Xtrain.reshape(Xtrain.shape[0], -1, 1)
#Xtrain = Xtrain.astype('float32')
print("X_train shape:", Xtrain.shape)
print(Xtrain.shape[0], "train samples")

print("Fetching Testing Data...")
data = np.asarray(list(csv.reader(open("./data/test.csv","rb"),delimiter=',')))
MNIST_data = np.delete(data,0,0)

Xtest = (np.array(MNIST_data[:,:], dtype=np.float32))
Xtest = np.reshape(Xtest,(Xtest.shape[0],1,28,28))
#Xtest = Xtest.reshape(Xtest.shape[0], -1, 1)
#X_test = X_test.astype('float32')
print(Xtest.shape[0], 'test samples')

#normalize X
Xtrain /= 255
Xtest /= 255

# Data Info, etc
nb_classes = 10
nb_epoch = 5000
batch_size = 128

poolingsize = 2
filter_size1 = 16
filter_size2 = 256
conv_kernal1 = 2
conv_kernal2 = 3
dropout_size = 0.4

hidden_units = 128

learning_rate = 1e-6
clip_norm = 1.0

# convert class vectors to binary class matrices
Ytrain = np_utils.to_categorical(Ytrain, nb_classes)

print(Xtrain.shape)
print('input_shape: ',Xtrain.shape[1:])
print(Ytrain.shape)

# Build model
model = Sequential()

# Modelled 99.99%
model.add(Convolution2D(filter_size1,conv_kernal1,conv_kernal1,border_mode='valid',input_shape=(1,28,28), activation="relu"))
model.add(Dropout(dropout_size))
#model.add(MaxPooling2D(pool_size=(poolingsize,poolingsize))) #M2
#model.add(Convolution2D(filter_size1,conv_kernal1,conv_kernal1,border_mode='valid', activation="relu"))
#model.add(Dropout(dropout_size))
#model.add(MaxPooling2D(pool_size=(poolingsize,poolingsize))) #M2

#
#model.add(Convolution2D(filter_size2,conv_kernal1,conv_kernal1,border_mode='valid', activation="relu")) #M2
#model.add(Dropout(dropout_size))
#model.add(Convolution2D(filter_size2,conv_kernal2,conv_kernal2,border_mode='valid', activation="relu")) #M2
#model.add(Dropout(dropout_size))
#model.add(MaxPooling2D(pool_size=(poolingsize,poolingsize))) #M2

#Try LSTM model
#model.add(LSTM(output_dim=hidden_units, init='glorot_uniform', inner_init='orthogonal', activation='relu', input_shape=Xtrain.shape[1:], return_sequences=False))
#, W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01), dropout_W=0.3, dropout_U=0.1))
# model.add(SimpleRNN(output_dim=hidden_units,
#                     init=lambda shape, name: normal(shape, scale=0.001, name=name),
#                     inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
#                     activation='relu',
#                     input_shape=Xtrain.shape[1:]))

model.add(Flatten())
#model.add(Dense(1024, activation="relu"))
#model.add(Activation('relu'))
#model.add(Dense(512, activation="relu")) #M2
#model.add(Dropout(0.25))
model.add(Dense(nb_classes, activation="softmax"))

model.summary()
#opt = 'adadelta'
#opt = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='acc', patience=100, verbose=0, mode='max')

#model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.01, shuffle=True, verbose=1, callbacks=[earlyStopping])

print(Ytrain.shape)

model.fit(Xtrain, Ytrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[earlyStopping])

score = model.evaluate(Xtrain, Ytrain, verbose=2)
print('Train score:', score[0])
print('Train accuracy:', score[1])

pred = model.predict(Xtest, batch_size=batch_size, verbose=0)

pred_val = np.argmax(pred,axis=1)

#image_id = (np.arange(pred_val.shape[0]))+1

#pred_arr = np.stack((image_id,pred_val),axis=1)
#pred_arr = np.reshape(pred_arr,(pred_val.shape[0]))

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
np.savetxt(
    './data/MNIST'+timestr+'.csv',
    np.c_[range(1,len(pred_val)+1), pred_val],
    delimiter=',',
    fmt='%d',
    header = 'ImageId,Label',
    comments = '',
    newline='\n'
)
