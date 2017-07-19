from os import listdir
from os.path import isfile, join
from scipy import misc
import pandas as pd
import numpy as np
import ast
from dateutil.parser import parse
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from dataset import read_dataset, read_hashtags

#Second approach
#Version with mean number of likes. To try the first approach without it
#set COLUMNS = 9 and remove meanNumberLikes from data list

COLUMNS = 10

print('Loading hashtags and dataset ...')
dataset, m = read_dataset()
hashtags = read_hashtags()

print('Loading image filenames...')
dir = 'images'
filenames = [f for f in listdir(dir) if isfile(join(dir, f))]

print('Building matrix...')
matrix = []
matrixFilenames = []

for index, row in dataset.iterrows():
	#Build filename from url
	filename = row.urlImage[row.urlImage.rfind('/')+1:]
	if not filename in filenames: continue	
	
	matrixFilenames.append(filename)
	nposts = row.numberPosts
	nfollowing = row.numberFollowing
	nfollowers = row.numberFollowers
	numberLikes = row.numberLikes
	date = parse(row.date)
	mydate = date.year * 365 + date.month * 30 + date.day
	mentions = len(ast.literal_eval(row.mentions))
	meanNumberLikes = m.loc[m.index == row.alias].numberLikes[0]

	#hashtag analysis
	tags = ast.literal_eval(row.tags)
	tags = [tag[1:] for tag in tags]
	tagsNumber = len(tags)
	tagsValues = [10000 - hashtags[tag] for tag in tags if tag in hashtags]
	tagsSum = sum(tagsValues)
	data = [nposts, nfollowing, nfollowers, mydate, date.weekday(), mentions, tagsNumber, tagsSum, meanNumberLikes, numberLikes]
	matrix.append(data)
	
print('Building model...')
TRAIN = 10016
matrix = np.array(matrix)
X = matrix[:,:COLUMNS-1]
y = matrix[:,COLUMNS-1]
X_train, X_test = np.split(X, [TRAIN])
y_train, y_test = np.split(y, [TRAIN])

IMAGE_SIZE = 256
CHANNELS = 3

#Convolutional neural network for images
image_input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS), name='image_input')
image_nn = Conv2D(4, (5,5), activation='relu')(image_input)
image_nn = Conv2D(16, (3,3), activation='relu')(image_input)
image_nn = MaxPooling2D(pool_size=(2,2))(image_nn)
image_nn = Conv2D(32, (3,3), activation='relu')(image_nn)
image_nn = Conv2D(64, (3,3), activation='relu')(image_nn)
image_nn = MaxPooling2D(pool_size=(2,2))(image_nn)
image_nn = Conv2D(128, (3,3), activation='relu')(image_nn)
image_nn = MaxPooling2D(pool_size=(2,2))(image_nn)
image_nn = Dropout(0.2)(image_nn)
image_nn = Conv2D(256, (3,3), activation='relu')(image_nn)
image_nn = MaxPooling2D(pool_size=(2,2))(image_nn)
image_nn = Dropout(0.2)(image_nn)
image_nn = Flatten()(image_nn)

#Neural network for user data and image metadata
data_input = Input(shape=(COLUMNS-1,), name='data_input')
data_nn = Dense(10, activation='relu')(data_input)

#Merge of the networks and last part of the network
output_nn = keras.layers.concatenate([image_nn, data_nn])
output_nn = Dense(500, activation='relu')(output_nn)
output_nn = Dropout(0.1)(output_nn)
output_nn = Dense(10, activation='relu')(output_nn)
output_nn = Dense(1, name='output')(output_nn)

#Model definition
model = Model(inputs=[image_input, data_input], outputs = [output_nn])

# Compile model
model.compile(loss='mape', optimizer='adam', metrics=['mape', 'mae', 'mse'])

def seqGenerator(X_train, y_train, batch_size, dir, filenames):
	index = 0
	num_train = X_train.shape[0]
	while 1:
		images_batch = []
		x_batch = []
		y_batch = []
		for i in range(batch_size):
			img = misc.imread(join(dir,filenames[index]), mode='RGB')
			img = misc.imresize(img, (IMAGE_SIZE,IMAGE_SIZE,CHANNELS))
			img = img / 255
			images_batch.append(img)
			x_batch.append(X_train[index])
			y_batch.append(y_train[index])
			index = index + 1 if not index == num_train - 1 else 0
		
		images_batch = np.asarray(images_batch)
		x_batch = np.asarray(x_batch)
		y_batch = np.asarray(y_batch)
		yield [images_batch, x_batch], y_batch
		

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

print('Training...')
# Fit the model
trainingset_size = len(X_train)
batch_size = 8
losshistory = LossHistory()

model.fit_generator(seqGenerator(X_train, y_train, batch_size, dir, matrixFilenames), steps_per_epoch=trainingset_size // batch_size, epochs=25, callbacks=[losshistory])

