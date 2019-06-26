#Author: Sebastian Hennig

import tensorflow as tf
import pickle, sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../Sebastian'))
import BuildVectors as bv
from bp_mll import bp_mll_loss
import random

f = pickle.load( open( "dataWlabelsAndDictsSplitPenREMOVED.pkl", "rb" ))

data = f['train']

BATCHSIZE = 1000
NN_BATCHSIZE = 3
EMBEDDINGSIZE = 256
HIDDENSIZE = 512
nbatches = len(data)//BATCHSIZE

random.shuffle(data)
wdfull = f['wordDictSmall']
labelDict = f['labelDict']
labelCount = f['labelCount']

batches = []
for i in range(0,nbatches):
	batch = data[i*BATCHSIZE:(i+1)*BATCHSIZE]
	batches += [batch]
	
if len(data)%BATCHSIZE != 0:
	last_batch = data[nbatches*BATCHSIZE:]
	batches += [last_batch]

def gen():
	for batch in batches:
		for b in batch:
			yield (bv.sequenceTranslate(b[0].split(),wdfull) ,bv.translate(b[1],labelDict), bv.translatePenalize(b[1],labelDict,labelCount))

ds = tf.data.Dataset.from_generator(
    gen, (tf.int64, tf.int64, tf.float32), (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))

#text = tf.placeholder(tf.int64, name='text')
#
#label = tf.placeholder(dtype=tf.int64, name='label')
#penalize = tf.placeholder(dtype=tf.float32, name='penalize')
#
#
#dataset = tf.data.Dataset.from_tensor_slices((text,label,penalize))

ds = ds.shuffle(20000).repeat().padded_batch(NN_BATCHSIZE,padded_shapes=([None],[None],[None]), padding_values=(tf.constant(-1, dtype=tf.int64) ,
	tf.constant(-1, dtype=tf.int64), tf.constant(0, dtype=tf.float32)))

iterator =  tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
dataset_init_op = iterator.make_initializer(ds, name='dataset_init')

next_batch = iterator.get_next()

text,label,penalize = next_batch

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
	#lengths_transposed = tf.expand_dims(lengths, 1)
	#ranges = tf.range(0,inp.shape[1],1,dtype=tf.int64)
	#range_row = tf.expand_dims(ranges, 0)
	#mask = tf.cast(tf.less(range_row, lengths_transposed), tf.int64)

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


loss = bp_mll_loss(tf.cast(label,tf.float32),LSTM(text),penalize)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# try it out
with tf.Session() as sess:
    i = 0
    sess.run(init_op)
    sess.run(dataset_init_op)

    while i < 25000:
    	_ , lols = sess.run([train_op,loss])
    	print(i)
    	print(lols)
    	if i % 1000 == 0:
    		save_path = saver.save(sess, "LSTMmodel")
    		print("Model saved in path: %s" % save_path)
    	i += 1
    save_path = saver.save(sess, "LSTMmodel")
    print("Model saved in path: %s" % save_path)

