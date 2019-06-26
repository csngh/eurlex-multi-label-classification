#Author: Sebastian Hennig

import pandas as pd
import pickle
import re

#Read the cleaned data from .csv as dataframe
df = pd.read_csv('imbalanced_labeled_duplicated.csv', header = None)

bigram_dict = {}
worddict = {}
print(len(df[0]))
for x in df[0]:
	words = x.split()
	words = [ y for y in words if re.search('^[a-z0-9]+$',y)]
	for i in range(0,len(words)-1):
		bigram = words[i]+' '+words[i+1]
		if bigram in bigram_dict:
			bigram_dict[bigram] += 1
		else:
			bigram_dict[bigram] = 1

	for word in words:
		if word in worddict:
			worddict[word] += 1
		else:
			worddict[word] = 1

print("Length Bigramdict: ", len(bigram_dict))

sorted_worddict = sorted(bigram_dict.items(), key=lambda kv: kv[1] , reverse=True)
sorted_word = sorted(worddict.items(), key=lambda kv: kv[1], reverse=True)
count = 0
for key,value in sorted_worddict:
	if value > 1:
		count += 1

print("1ns: ",len(bigram_dict)-count)

for x in sorted_worddict[:10]:
	print(x)

small_bi_dict = [x for x in sorted_worddict if x[1] > 1]

small_worddict = [x for x in sorted_word if x[1] > 1]

print("Small BiDict Lenght: ", len(small_bi_dict))

data = {}

data['RawData'] = list(zip(df[0],df[1]))
data['BiDict'] = small_bi_dict
data['worddict'] = worddict
data['worddictS'] = small_worddict

pickle_out = open("dictsAndData.pkl","wb")
pickle.dump(data, pickle_out)
pickle_out.close()
