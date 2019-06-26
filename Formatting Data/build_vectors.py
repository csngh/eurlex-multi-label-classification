#Author: Sebastian Hennig

import pickle
import numpy as np

def translate(inp, dic, invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	vec = np.zeros(len(dic))
	for x in inp:
		if x in dic:
			vec[dic[x]] += 1
	return vec


def translatePenalize(inp, dic, dic2, invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	vec = np.ones(len(dic))
	for x in inp:
		if x in dic:
			vec[dic[x]] = dic2[x]
	return vec


def sequenceTranslate(inp,dic,invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	vec = []
	for x in inp:
		if x in dic:
			vec.append(dic[x])
	return np.array(vec)


def translateNGram(inp,dic, n,invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	vec = np.zeros(len(dic))
	for i in range(0,len(inp)-n+1):
		ngram = ""
		for x in range(0,n):
			ngram += inp[i+x]+" "
		ngram = ngram.rstrip()
		if ngram in dic:
			vec[dic[ngram]] += 1
	return vec


def tfIdfTranslate(inp, dic, idf, invert_dict=False):
	if invert_dict:
		dic = {v: k for k, v in dic.items()}
	tf = {}
	for word in inp:
		if word in dic:
			if word in tf:
				tf[word] += 1
			else:
				tf[word] = 1
	vec = np.zeros(len(dic))

	for t in tf:
		vec[dic[t]] = tf[t]/idf[t]
	return vec
