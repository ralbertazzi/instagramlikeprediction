import pandas as pd

def read_dataset():
	dataset = pd.read_csv('datasets/dataset.csv', quotechar='"', skipinitialspace=True,encoding='utf-8-sig')
	dataset = dataset.loc[dataset['multipleImage'] == False]
	dataset = dataset.loc[dataset['numberFollowers'] < 1000000]
	dataset = dataset.loc[dataset['numberLikes'] > 0]
	m = dataset.groupby('alias').mean()
	
	return dataset, m

def read_hashtags():
	with open('datasets/hashtags.txt', 'r') as file:
		hashtags = file.read().split()
		hashtags = dict(enumerate(hashtags))
		hashtags = {v:k for k,v in hashtags.items()}
		return hashtags
