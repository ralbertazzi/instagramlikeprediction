from os import listdir
from os.path import isfile, join
from scipy import misc
import pandas as pd
import numpy as np
import ast
import operator
from dateutil.parser import parse
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from features import LabelTrainer
from dataset import read_dataset

#Fourth approach with VGG-19 features

dir = 'images'

print('Loading VGG19 model...')
base_model = VGG19()
vgg19model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

print('Loading datasets and pre-processing...')
dataset, m = read_dataset()

filenames = [f for f in listdir(dir) if isfile(join(dir, f))]

positivitymatrix = []
positivity = []

print('Building matrix...')
for index, row in dataset.iterrows():
	filename = row.urlImage[row.urlImage.rfind('/')+1:]
	if index % 1000 == 0: print(index)
	if not filename in filenames: continue

	#see tutorial on keras
	img = image.load_img(join(dir, filename), target_size=(224,224))
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)
	vgg19features = vgg19model.predict(x)
	
	numberLikes = row.numberLikes
	meanNumberLikes = m.loc[m.index == row.alias].numberLikes[0]
	pos = numberLikes/meanNumberLikes

	positivitymatrix.append(vgg19features)
	positivity.append(pos)

print('Training set: ', len(positivitymatrix))
positivitymatrix = np.array(positivitymatrix)
TRAIN = 8000

pos_x_train, pos_x_test = np.split(positivitymatrix, [TRAIN])
pos_y_train, pos_y_test = np.split(positivity, [TRAIN])

def train_vgg19features():
	model.add(Dense(2000, input_dim=positivitymatrix.shape[1], activation='relu'))
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mape', optimizer='adam', metrics=['mae', 'mse', 'mape'])
	model.fit(pos_x_train, pos_y_train, epochs=100, batch_size=8, validation_data=(pos_x_test, pos_y_test))

print('Building model...')
model = Sequential()
train_vgg19features()

