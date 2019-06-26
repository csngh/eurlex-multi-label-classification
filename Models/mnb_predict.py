#Author: Sebastian Hennig

from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle, sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../Sebastian'))
import BuildVectors as bv
import numpy as np
from sklearn.metrics import hamming_loss,zero_one_loss,jaccard_score, accuracy_score,f1_score

BATCHSIZE = 1000

f = pickle.load( open( "dataWlabelsAndDictsSplitPenREMOVED.pkl", "rb" ))
models = joblib.load( open( "MNBModelsDown.pkl", "rb" ))
idf = pickle.load( open( "IDFRemoved.pkl", "rb" ))

dat = f['test']
idfDict = idf['idf']
wordDict = f['wordDictSmall']
BATCHSIZE = 1000
labelDict = f['labelDict']


def gen(batches):
	for batch in batches:
		yield [ [bv.tfIdfTranslate(x.split(),wordDict,idfDict),bv.translate(y,labelDict)] for x,y in batch]


nbatches = len(dat)//BATCHSIZE

batchs = []
for i in range(0,nbatches):
	batch = dat[i*BATCHSIZE:(i+1)*BATCHSIZE]
	batchs += [batch]
	

if len(dat)%BATCHSIZE != 0:
	last_batch = dat[nbatches*BATCHSIZE:]
	batchs += [last_batch]


pred = np.array([], dtype=np.int64).reshape(0,len(labelDict))
true = np.array([], dtype=np.int64).reshape(0,len(labelDict))

for b in gen(batchs):
	X,y = list(zip(*b))
	predL = []
	for _, v in models.items():
		if v == 'undefined':
			predL.append([0]*BATCHSIZE)
		else:
			predL.append(v.predict(X))
	preds = np.transpose(np.array(predL))
	truths = np.array(y)
	pred = np.vstack([pred,preds])
	true = np.vstack([true,truths])
	print(pred.shape,true.shape)

print('Hamming Loss:', hamming_loss(pred,true) )
print('Zero One Loss:', zero_one_loss(pred,true) )
print('Jaccard Score:', jaccard_score(pred,true,average='samples') )
print('F1-Score Micro:', f1_score(pred,true,average='micro') )
print('F1-Score Macro:', f1_score(pred,true,average='macro') )
print('Accuracy :', accuracy_score(pred,true) )
	

