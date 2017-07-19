import pandas as pd
import urllib.request
import ast

#Script to download images from dataset.csv

dataset = pd.read_csv('dataset.csv', quotechar='"', skipinitialspace=True)
rows = dataset.shape[0]

for row in range(rows):
	print('downloading row ', row, '\r')
	url = dataset.urlImage[row]

	#multiple images
	if url.startswith('['):
		urls = ast.literal_eval(url)
	else:
		urls = [url]
	
	filenames = [url[url.rfind('/') + 1:] for url in urls]
	
	for url, filename in zip(urls, filenames):
		try:
			urllib.request.urlretrieve(url, 'images/' + filename)
		except:
			print('Error downloading image(s) at row ', row)
			with open('log.txt', 'a') as log:
				log.write('Error row ' + str(row))
