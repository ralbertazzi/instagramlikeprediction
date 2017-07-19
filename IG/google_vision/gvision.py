from os import listdir
from os.path import isfile, join
import json
from google.cloud import vision
from google.cloud.vision.feature import Feature, FeatureTypes

#Script to perform label detection of images using Google Vision API

dir = 'images'
filenames = [f for f in listdir(dir) if isfile(join(dir, f))]

BATCH = 5
client = vision.Client('InstagramLabels')
output = []

for index in range(0,len(filenames),BATCH):
	print('index ', index)

	batch = client.batch()
	files = filenames[index:index+BATCH-1]
	images = [client.image(filename=join(dir,img)) for img in files]
	features = [Feature(FeatureTypes.FACE_DETECTION,10), Feature(FeatureTypes.LABEL_DETECTION,20)]
	for image in images:
		batch.add_image(image, features)

	results = batch.detect()
	for result, filename in zip(results,files):
		numFaces = result.faces
		labels = [{label.description: label.score} for label in result.labels]
		labels = {k: v for d in labels for k, v in d.items()}
		v = {'filename': filename, 'numFaces': len(result.faces), 'labels': labels}
		output.append(v)

with open('gvision3.json', 'w') as outputjson:
	outputjson.write(json.dumps(output))
