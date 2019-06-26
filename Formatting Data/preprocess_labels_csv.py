#Author: Sebastian Hennig

import pickle
import math
import random

data =  pickle.load( open( "dictsAndDataRemoved.pkl", "rb" ))

labels = {}
i = 0
labelDict = {}
lableCount = {}
for x in data['RawData']:
	temp = x[1].split(", ")
	for y in temp:
		if y not in labelDict:
			labelDict[y] = i
			i = i+1
		if y not in lableCount:
			lableCount[y] = 1
		else:
			lableCount[y] += 1

for k,v in lableCount.items():
	lableCount[k] = 1/v


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

DataPoints = [[x[0],x[1].split(", ")] for x in data['RawData']]

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

pickle_out = open("dataWlabelsAndDictsSplitPenREMOVED.pkl","wb")
pickle.dump(saveFile, pickle_out)
pickle_out.close()


