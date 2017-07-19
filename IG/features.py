import ast
import numpy as np
from os.path import join
from collections import Counter
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras import regularizers

class LabelTrainer:

	def read_features(self, filename='gvision10.json', common_labels_min = 10):
		print('Reading labels and building training set...')
		with open(join('google_vision',filename), 'r') as gvision:
			self.all_features = ast.literal_eval(gvision.read())
			self.filenames = [v['filename'] for v in self.all_features]

		c = Counter([k for v in self.all_features for k in v['labels'].keys()])
		common_labels = [key for (key,count) in c.most_common() if count >= common_labels_min]

		trainingset_size, self.row_size = len(self.all_features), len(common_labels)
		matrix = np.zeros((trainingset_size, self.row_size))

		for row, image in enumerate(self.all_features):
			for key in image['labels'].keys():
				if key in common_labels:
					idx = common_labels.index(key)
					matrix[row,idx] = 1.0
					#matrix[row,idx] = image['labels'][key]
		
		self.common_labels = common_labels
		self.train_size = int(trainingset_size * 0.8)
		self.x_train, self.x_test = np.split(matrix, [self.train_size])
		self.matrix = matrix

	def build_model(self, encoding_dim = 50):
		print('Building labels encoder and autoencoder...')
		input_layer = Input(shape=(self.row_size,))
		encoded = Dense(encoding_dim, activation='relu')(input_layer)
		decoded = Dense(self.row_size, activation='sigmoid')(encoded)

		self.autoencoder = Model(input_layer, decoded)
		self.encoder = Model(input_layer, encoded)

		self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
		self.encoding_dim = encoding_dim
	
	def train(self, epch=50, batch=32):
		print('Training labels autoencoder...')
		self.autoencoder.fit(self.x_train, self.x_train, epochs=epch, batch_size=batch, shuffle=True, validation_data=(self.x_test, self.x_test))

	def eval(self, test=1):
		print('evaluating label encoder...')
		#returns the index of the values > thresh into the row array
		def get_values(row, thresh=0.5):
			return [i for (i,x) in enumerate(row) if x >= thresh]

		x_in = self.x_test if test else self.x_train
		count = 0
		x_predict = self.autoencoder.predict(x_in)
		for x,y in zip(x_in,x_predict):
			count = count + len(set(get_values(x)).symmetric_difference(set(get_values(y))))
		print(count)
		print(count/len(x_in))
	
	def get_features(self, filename):
		try:
			return next((o for o in self.all_features if o['filename'] == filename))
		except:
			return None
	
	def get_data(self, filename):
		row,v = next(((i,v) for (i,v) in enumerate(self.all_features) if v['filename'] == filename))
		numFaces = v['numFaces']
		input_row = self.matrix[row:row+1]
		encoded = self.encoder.predict(input_row)
		encoded = encoded[0]
		encoded.shape = (1, self.encoding_dim)
		encoded = encoded[0]
		return numFaces, encoded

	def get_label_input_vector(self, filename):
		row,v = next(((i,v) for (i,v) in enumerate(self.all_features) if v['filename'] == filename))
		input_row = self.matrix[row:row+1]
		return input_row

	def predict_data(self, labels):
		matrix = np.zeros((1, self.row_size))
		for key in labels.keys():
			if key in self.common_labels:
				idx = self.common_labels.index(key)
				matrix[row,idx] = 1.0
				#matrix[0,idx] = labels[key]

		encoded = self.encoder.predict(matrix)
		return encoded
	
	def predict_data_from_input_vector(self, labels):
		return self.encoder.predict(labels)

if __name__ == "__main__":
	l = LabelTrainer()
	l.read_features()
	l.build_model()
	l.train()
	print(l.get_data(l.all_features[1000]['filename']))
	
	l.eval()
	l.eval(0)

