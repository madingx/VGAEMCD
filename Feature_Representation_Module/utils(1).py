"""
Author: Maosen Li, Shanghai Jiao Tong University
"""

import tensorflow as  tf
import numpy as np

def unif_weight_init(shape, name=None):

    initial = tf.compat.v1.random_uniform(shape, minval=-np.sqrt(6.0/(shape[0]+shape[1])), maxval=np.sqrt(6.0/(shape[0]+shape[1])), dtype=tf.float32)

    return tf.Variable(initial, name=name)


def sample_gaussian(mean, diag_cov):

    z = mean+tf.compat.v1.random_normal(tf.shape(diag_cov))*diag_cov

    return z


def sample_gaussian_np(mean, diag_cov):

    z = mean+np.random.normal(size=diag_cov.shape)*diag_cov
    
    return z


def gcn_layer_id(norm_adj_mat, W, b):

    return tf.nn.relu(tf.add(tf.compat.v1.sparse_tensor_dense_matmul(norm_adj_mat, W), b))


def gcn_layer(norm_adj_mat, h, W, b,zy):
    a=tf.add(tf.matmul(tf.compat.v1.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b)
    ldg = tf.nn.sigmoid(tf.matmul(a, a, transpose_b=True))
    attentions = tf.nn.softmax(zy * ldg)
    ww=tf.matmul(attentions, a)
    return ww
def gcn_layer1(norm_adj_mat, h, W, b):
    a=tf.add(tf.matmul(tf.compat.v1.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b)

    return a

def sigmoid(x):

    return 1.0/(1.0+np.exp(-x))
