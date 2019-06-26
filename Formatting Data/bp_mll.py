#Author: https://github.com/vanHavel/bp-mll-tensorflow

import numpy as np
import tensorflow as tf

# bp mll loss function
# y_true, y_pred must be 2D tensors of shape (batch dimension, number of labels)
# y_true must satisfy y_true[i][j] == 1 iff sample i has label j
def bp_mll_loss(y_true, y_pred,penalize):

    # get true and false labels
    shape = tf.shape(y_true)
    y_i = tf.equal(y_true, tf.ones(shape))
    y_i_bar = tf.not_equal(y_true, tf.ones(shape))

    # get indices to check
    truth_matrix = tf.to_float(pairwise_and(y_i, y_i_bar))
    

    truth_matrix = tf.multiply(truth_matrix, tf.expand_dims(penalize,2))

    # calculate all exp'd differences
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = tf.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1,2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_i_bar), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    results = tf.divide(sums, normalizers)

    # average error
    return tf.reduce_mean(results)

# compute pairwise differences between elements of the tensors a and b
def pairwise_sub(a, b):
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.subtract(column, row)

# compute pairwise logical and between elements of the tensors a and b
def pairwise_and(a, b):
    column = tf.expand_dims(a, 2)
    row = tf.expand_dims(b, 1)
    return tf.logical_and(column, row)
