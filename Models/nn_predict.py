#Author: Sebastian Hennig

import tensorflow as tf
import pickle, sys, os
import numpy as np
from sklearn.metrics import hamming_loss,zero_one_loss,jaccard_score, accuracy_score,f1_score

sys.path.insert(1, os.path.join(sys.path[0], '../Sebastian'))
import BuildVectors as bv

f = pickle.load( open( "dataWlabelsAndDictsSplitPen.pkl", "rb" ))

data = f['test']
wdfull = f['wordDictSmall']
labelDict = f['labelDict']

text = tf.placeholder(tf.int64,shape=[None,None],name='text')

#testi = np.array(bv.sequenceTranslate(data[0][0].split(),wdfull))

NN_BATCHSIZE = 4
EMBEDDINGSIZE = 256
HIDDENSIZE = 512


Embedding = tf.Variable(tf.random_uniform(
    [len(wdfull)+1,EMBEDDINGSIZE],
    minval=-0.2,
    maxval=0.2,
    dtype=tf.float32),name='emb')

#For Softmax
weight = tf.Variable(tf.truncated_normal([HIDDENSIZE, len(labelDict)], stddev=0.01),name='w1')
bias = tf.Variable(tf.constant(0.1, shape=[len(labelDict)]),name='b1')


def LSTM(inp):

	lengths = tf.math.count_nonzero(inp+1,1)

	inp = tf.nn.embedding_lookup(Embedding,inp)
	output, _ = tf.nn.dynamic_rnn(
	    tf.contrib.rnn.LSTMBlockCell(HIDDENSIZE,name='rnn'),
	    inp,
	    dtype=tf.float32,
	    sequence_length=lengths)
	lr = last_relevant(output,lengths)

	#prediction = tf.nn.softmax(tf.matmul(lr, weight) + bias)
	prediction = tf.nn.tanh(tf.matmul(lr, weight) + bias,name='prediction')

	return prediction


def last_relevant(output, length):
	batch_size = tf.shape(output)[0]
	max_length = tf.shape(output)[1]
	out_size = int(output.get_shape()[2])
	index = tf.range(0, batch_size) * max_length + tf.cast((length - 1),tf.int32)
	flat = tf.reshape(output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return relevant

prediction = LSTM(text)

saver = tf.train.Saver()

with tf.Session() as sess:

	saver.restore(sess, tf.train.latest_checkpoint('models/UpSampling/'))

	graph = tf.get_default_graph()

	pred = sess.run(prediction,feed_dict={text: [bv.sequenceTranslate(data[0][0].split(),wdfull)]})

	pred = np.array(pred)
	indices = pred > 0
	comp = np.zeros(pred.shape)
	comp[indices] = 1


	for x in range(1,len(data)):
		pred = sess.run(prediction,feed_dict={text: [bv.sequenceTranslate(data[0][0].split(),wdfull)]})


		pred = np.array(pred)

		indices = pred > 0
		  
		tmp = np.zeros(pred.shape)

		tmp[indices] = 1

		comp = np.concatenate((comp,tmp))

	truths = np.array([bv.translate(b[1],labelDict) for b in data])
	
	print('Hamming Loss:', hamming_loss(comp,truths) )
	print('Zero One Loss:', zero_one_loss(comp,truths) )
	print('Jaccard Score:', jaccard_score(comp,truths,average='samples') )
	print('F1-Score Micro:', f1_score(comp,truths,average='micro') )
	print('F1-Score Macro:', f1_score(comp,truths,average='macro') )
	print('Accuracy :', accuracy_score(comp,truths) )
	