# -*- coding: utf-8 -*=
import numpy as np

import os
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.append('/home/fanyixing/MatchZoo/matchzoo/layers')
from Cropping2D import Cropping2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()# path='/home/fanyixing/TF/data/MNIST')

print x_train.shape

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
print x_train.shape
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print y_train.shape
print y_train[:10]

y_train = np_utils.to_categorical(y_train, 10)
y_test= np_utils.to_categorical(y_test, 10)
print y_train.shape

model = Sequential()
model.add(Cropping2D(cropping=((1,1), (1,1)), dim_ordering='th', input_shape=(1, 28, 28)))
print model.output_shape
model.add(Conv2D(32, (2, 2), padding='same', activation='relu' ))
print model.output_shape


