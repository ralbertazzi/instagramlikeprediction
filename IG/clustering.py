from features import LabelTrainer
from sklearn.cluster import KMeans
from scipy import misc
from os.path import join
import numpy as np
import pandas as pd
from dataset import read_dataset

class Clusterer:

	def cluster(self):
		labelTrainer = LabelTrainer()
		labelTrainer.read_features()
		labelTrainer.build_model(encoding_dim=100)
		labelTrainer.train(epch=15)

		print('Building matrix...')
		X = []
		for filename in labelTrainer.filenames:
			_, encodedLabel = labelTrainer.get_data(filename)
			X.append(encodedLabel)
		X = np.array(X)

		print('Clustering...')
		kmeans = KMeans(n_clusters=50, random_state=0).fit(X)
		predictions = kmeans.predict(X)

		dictClassFiles = {}
		dictFileClass = {}
		for filename, pred in zip(labelTrainer.filenames, predictions):
			if pred in dictClassFiles:
				dictClassFiles[pred].append(filename)
			else:
				dictClassFiles[pred] = [filename]

			dictFileClass[filename] = {'class': pred}


		print('Loading datasets and pre-processing...')
		dataset, m = read_dataset()

		userGood = {}
		
		print('Building user performance metrics...')
		for index, row in dataset.iterrows():
			filename = row.urlImage[row.urlImage.rfind('/')+1:]
			if index % 1000 == 0: print(index)
			if not filename in labelTrainer.filenames: continue
			numberLikes = row.numberLikes
			meanNumberLikes = m.loc[m.index == row.alias].numberLikes[0]
			pos = numberLikes/meanNumberLikes
			dictFileClass[filename]['likes'] = numberLikes
			dictFileClass[filename]['mean'] = meanNumberLikes
			dictFileClass[filename]['pos'] = pos
			cl = dictFileClass[filename]['class']
	
			if not row.alias in userGood:
				userGood[row.alias] = [(cl,pos)]
			else:
				userGood[row.alias].append((cl,pos))

		#Read 5 images for each cluster in the list
		#for i in [0,1,10,16,38]:
		#	images = dictClassFiles[i][:5]
		#	for image in images:
		#		img = misc.imread(join('images', image))
		#		misc.imshow(img)
		
		alpha = 0.5
		userArrays = {}
		for user in userGood:
			userClasses = list(set((cl for (cl,pos) in userGood[user])))
			arr = np.zeros(50)
			userGoodRestricted = userGood[user][::2]
			for userClass in userClasses:
				s = [pos-1 for (cl,pos) in userGoodRestricted if cl == userClass]
				l = len(s)
				if l > 0:
					s = sum(s) / l
				else:
					s = 0
				arr[userClass] = s

			for i in range(len(arr)):
				if arr[i] > alpha:
					arr[i] = 1
				elif arr[i] < -alpha:
					arr[i] = -1
				else:
					arr[i] = 0

			userArrays[user] = arr

		self.userArrays = userArrays
		self.dictFileClass = dictFileClass

	def predictPos(self, user, filename):
		cl = self.dictFileClass[filename]['class']
		return self.userArrays[user][cl]
	
