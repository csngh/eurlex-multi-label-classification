#Author: Sebastian Hennig

from sklearn.naive_bayes import MultinomialNB
import pickle, sys, os
from sklearn.externals import joblib
sys.path.insert(1, os.path.join(sys.path[0], '../Sebastian'))
import BuildVectors as bv
import itertools
import random

BATCHSIZE = 1000

f = pickle.load( open( "dataWlabelsAndDictsSplitPenREMOVED.pkl", "rb" ))
idf = pickle.load( open( "IDFRemoved.pkl", "rb" ))

dat = f['train']+f['valid']
idfDict = idf['idf']
wordDict = f['wordDictSmall']
labelDict = f['labelDict']


def gen(batches):
	for batch in batches:
		yield [ [bv.tfIdfTranslate(x.split(),wordDict,idfDict),y] for x,y in batch]

models = {}
l = 0
for k,v in labelDict.items():
	#deeeeep copy
	data = [i for i in dat]
	l += 1

	labelData = []
	for d in data:
		x,y = d
		added = False
		if k in d[1]:
			labelData += [[x,1]]
			data.remove(d)
	#278 labels that are not contained in train set.
	if not labelData:
		models[k] = 'undefined'
		continue

	#change between All other are negative(False) and only proportional amount is negative (True) 
	if False:
		numNegSamples = 12 * len(labelData)
		negativeSamples = [data.pop(random.randrange(len(data))) for _ in range(numNegSamples)]
		negativeSamples = [[x,y] for x, y in negativeSamples if k not in y]
		while len(negativeSamples) != numNegSamples:
			negativeSamples += [data.pop(random.randrange(len(data))) for _ in range(numNegSamples-len(negativeSamples))]
			negativeSamples = [[x,y] for x, y in negativeSamples if k not in y]
	else:
		negativeSamples = data

	negativeSamples = [[x,0] for x, _ in negativeSamples]

	labelData += negativeSamples

	nbatches = len(labelData)//BATCHSIZE

	batches = []
	for i in range(0,nbatches):
		batch = labelData[i*BATCHSIZE:(i+1)*BATCHSIZE]
		batches += [batch]
		
	if len(labelData)%BATCHSIZE != 0:
		last_batch = labelData[nbatches*BATCHSIZE:]
		batches += [last_batch]


	clf = MultinomialNB()
	for b in gen(batches):
		X,Y = list(zip(*b))
		clf.partial_fit(X,Y,[0,1])
	models[k] = clf
	print('Trained Model ',l)



pickle_out = open("MNBModelsDown.pkl","wb")
joblib.dump(models, pickle_out)
pickle_out.close()


