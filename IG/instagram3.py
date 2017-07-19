from os import listdir
from os.path import isfile, join
from scipy import misc
import pandas as pd
import numpy as np
import ast
import operator
from dateutil.parser import parse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from dataset import read_dataset
from features import LabelTrainer

#Third approach and fourth approach with custom CNN

dir = 'images'

print('Loading datasets and pre-processing...')
dataset, m = read_dataset()

labelTrainer = LabelTrainer()
labelTrainer.read_features()
labelTrainer.build_model(encoding_dim=100)
labelTrainer.train(epch=15)

matrixFilenames = []

positivitymatrix = []
positivity = []

print('Building matrix...')
for index, row in dataset.iterrows():
	filename = row.urlImage[row.urlImage.rfind('/')+1:]
	if index % 1000 == 0: print(index)
	if not filename in labelTrainer.filenames: continue

	matrixFilenames.append(filename)
	numberLikes = row.numberLikes
	meanNumberLikes = m.loc[m.index == row.alias].numberLikes[0]

	_, encodedLabel = labelTrainer.get_data(filename)

	pos = numberLikes/meanNumberLikes

	positivitymatrix.append(encodedLabel.tolist())
	positivity.append(pos)


print('Training set: ', len(positivitymatrix))
TRAIN = 8000

positivitymatrix = np.array(positivitymatrix)
pos_x_train, pos_x_test = np.split(positivitymatrix, [TRAIN])
pos_y_train, pos_y_test = np.split(positivity, [TRAIN])

IMAGE_SIZE = 256
CHANNELS = 3

#Input generator for CNN
def seqGenerator(batch_size, dir, filenames, mode):
	files = filenames[:TRAIN] if mode == "train" else filenames[TRAIN:]
	print(len(files))
	pos = pos_y_train if mode == "train" else pos_y_test
	index = 0
	while 1:
		images_batch = []
		y_batch = []
		for i in range(batch_size):
			img = misc.imread(join(dir,files[index]), mode='RGB')
			img = misc.imresize(img, (IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
			img = img / 255
			images_batch.append(img)
			y_batch.append(pos[index])
			index = index + 1 if not index == len(files) - 1 else 0
		
		images_batch = np.asarray(images_batch)
		y_batch = np.asarray(y_batch)
		yield images_batch, y_batch

#Third approach
def train_labels():
	model.add(Dense(200, input_dim=positivitymatrix.shape[1], activation='relu', activity_regularizer=regularizers.l1(0.01)))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu', activity_regularizer=regularizers.l1(0.01)))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mape', optimizer='adam', metrics=['mae', 'mse', 'mape'])
	model.fit(pos_x_train, pos_y_train, epochs=1000, batch_size=8, validation_data=(pos_x_test, pos_y_test))

#Fourth approach with custom CNN
def train_images():
	model.add(Conv2D(96, (11,11), strides=4, input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=3, strides=2))
	model.add(BatchNormalization())
	model.add(Conv2D(256, (5,5), strides=2, activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=3, strides=2))
	model.add(BatchNormalization())
	model.add(Conv2D(384, (3,3), activation='relu'))
	model.add(Conv2D(256, (3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=3, strides=2))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(1))

	batch_size = 8
	model.compile(loss='mape', optimizer='adam', metrics=['mape', 'mae', 'mse'])
	model.fit_generator(seqGenerator(batch_size, dir, matrixFilenames, mode='train'), steps_per_epoch = TRAIN // batch_size, epochs=25, validation_data = seqGenerator(1, dir, matrixFilenames, mode='test'), validation_steps = len(positivity) - TRAIN)

print('Building model...')
model = Sequential()
train_images()

