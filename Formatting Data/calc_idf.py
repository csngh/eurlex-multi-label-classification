#Author: Sebastian Hennig

from multiprocessing import Pool
import pickle
import math

data =  pickle.load( open( "dataWlabelsAndDictsSplitPenREMOVED.pkl", "rb" ))

wDFull = data['wordDictFull']

DataPoints = data['completeData']

wdfList = list(wDFull.keys())




def countfq(word):
	fq = 0
	for txt, _ in DataPoints:
		if word in txt:
			fq += 1
	return fq


if __name__ == '__main__':
	p = Pool()
	counts = p.map(countfq,wdfList)
	print(len(counts))
	print(len(wdfList))
	###Calc IDF
	cdict = dict(zip(wdfList,counts))
	idf = {}
	for x in wdfList:
		if cdict[x] == 0:
			idf[x] = 0
		else:
			idf[x] = math.log(len(DataPoints)/cdict[x])
	saveFile = {}
	saveFile['counts'] = cdict
	saveFile['idf'] = idf
	pickle_out = open("IDFRemoved.pkl","wb")
	pickle.dump(saveFile, pickle_out)
	pickle_out.close()
