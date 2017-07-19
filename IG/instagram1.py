import pandas as pd
import numpy as np
import ast
from dataset import read_dataset, read_hashtags
from dateutil.parser import parse
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Version with mean number of likes. To try the first approach without it
#set COLUMNS = 9 and remove meanNumberLikes from data list

COLUMNS = 10

dataset, m = read_dataset()
hashtags = read_hashtags()

matrix = []

print('Building matrix...')
for index, row in dataset.iterrows():
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
TRAIN = 10000
matrix = np.array(matrix)
X = matrix[:,:COLUMNS-1]
y = matrix[:,COLUMNS-1]
X_train, X_test = np.split(X, [TRAIN])
y_train, y_test = np.split(y, [TRAIN])

model = Sequential()
model.add(Dense(80, input_dim=COLUMNS-1, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mape', optimizer='adam', metrics=['mae', 'mse'])

# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test,y_test))

