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
from features import LabelTrainer
from clustering import Clusterer
from dataset import read_dataset

#Fifth approach

dir = 'images'

c = Clusterer()
c.cluster()

print('Loading datasets and pre-processing...')
dataset, m = read_dataset()

labelTrainer = LabelTrainer()
labelTrainer.read_features()
labelTrainer.build_model(encoding_dim=100)
labelTrainer.train(epch=15)

filenames = labelTrainer.filenames
matrixFilenames = []

positivitymatrix = []
positivity = []

print('Building matrix...')
for index, row in dataset.iterrows():
	filename = row.urlImage[row.urlImage.rfind('/')+1:]
	if index % 1000 == 0: print(index)
	if not filename in filenames: continue

	matrixFilenames.append(filename)
	numberLikes = row.numberLikes
	meanNumberLikes = m.loc[m.index == row.alias].numberLikes[0]

	_, encodedLabel = labelTrainer.get_data(filename)
	clusterPerformance = c.predictPos(row.alias, filename)

	pos = numberLikes/meanNumberLikes

	encodedLabelList = encodedLabel.tolist()
	#Append cluster performance
	encodedLabelList.append(clusterPerformance)

	positivitymatrix.append(encodedLabelList)
	positivity.append(pos)

print('Training set: ', len(positivitymatrix))
TRAIN = 8000

positivitymatrix = np.array(positivitymatrix)

pos_x_train, pos_x_test = np.split(positivitymatrix, [TRAIN])
pos_y_train, pos_y_test = np.split(positivity, [TRAIN])

def train_labels():
	model.add(Dense(200, input_dim=positivitymatrix.shape[1], activation='relu', activity_regularizer=regularizers.l1(0.01)))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu', activity_regularizer=regularizers.l1(0.01)))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(1))

	model.compile(loss='mape', optimizer='adam', metrics=['mae', 'mse', 'mape'])
	model.fit(pos_x_train, pos_y_train, epochs=100, batch_size=8, validation_data=(pos_x_test, pos_y_test))

print('Building model...')
model = Sequential()
train_labels()

