#Author: Sebastian Hennig

import pickle
import math
import random

labelFile = open('id2class_eurlex_eurovoc.qrels','r')

labels = {}
i = 0
labelDict = {}
lableCount = {}
for x in labelFile:
	temp = x.split()

	if temp[1] in labels:
		labels[temp[1]].append(temp[0])
	else:
		labels[temp[1]] = [temp[0]]

	if temp[0] not in labelDict:
		labelDict[temp[0]] = i
		i = i+1
	if temp[0] not in lableCount:
		lableCount[temp[0]] = 1
	else:
		lableCount[temp[0]] += 1

for k,v in lableCount.items():
	lableCount[k] = 1/v


data =  pickle.load( open( "dictsAndData.pkl", "rb" ))
docs = data['RawData']
biDict = data['BiDict']


properBiDict = {}
i = 0
for x in biDict:
	if x[0] not in properBiDict:
		properBiDict[x[0]] = i
		i +=1


wDFull = data['worddict']
properFullDict = {}

i=0
for x in wDFull:
	if x not in properFullDict:
		properFullDict[x] = i
		i +=1

wDSmall = data['worddictS']
properSmallDict = {}
i = 0
for x in wDSmall:
	if x[0] not in properSmallDict:
		properSmallDict[x[0]] = i
		i +=1


DataPoints = []
for doc in docs:
	id_d = str(int(doc[0]))
	if id_d in labels:
		datapoint = [doc[1],labels[id_d]]
		DataPoints.append(datapoint)
	else:
		print("ID ",id_d," has no LABELS")

#Divide in Training Test Valid
random.shuffle(DataPoints)

train = 70
valid = 10
test = 20

ntrain = round(len(DataPoints)*(train/100))
nvalid = round(len(DataPoints)*(valid/100))

TrainSet = []
while ntrain > 0:
	ntrain -= 1
	TrainSet.append(DataPoints.pop(random.randrange(len(DataPoints))))

ValidSet = []
while nvalid > 0:
	nvalid -= 1
	ValidSet.append(DataPoints.pop(random.randrange(len(DataPoints))))

TestSet = DataPoints

print(len(TrainSet))
print(len(ValidSet))
print(len(TestSet))

saveFile = {}

saveFile['train'] = TrainSet
saveFile['valid'] = ValidSet
saveFile['test'] = TestSet
saveFile['labelDict'] = labelDict
saveFile['labelCount'] = lableCount
saveFile['biDict'] = properBiDict
saveFile['wordDictFull'] = properFullDict
saveFile['wordDictSmall'] = properSmallDict
saveFile['freqcountWords'] = wDFull
saveFile['completeData'] = TrainSet+ValidSet+TestSet

pickle_out = open("dataWlabelsAndDictsSplitPen.pkl","wb")
pickle.dump(saveFile, pickle_out)
pickle_out.close()


