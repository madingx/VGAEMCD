import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from model import Feature_representation_module as VGAE
import networkx as nx

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='./data', help='path of the dataset')
parser.add_argument('--dataset_name', dest='dataset_name', default='citation', help='name of the dataset')
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='result of the model testing')

parser.add_argument('--n_hidden', dest='n_hidden', type=int, default=200, help='dimension of hidden layer')
parser.add_argument('--dropout', dest='dropout', type=bool, default=True, help='Using dropout in training')
parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=0.5, help='prob of keeping activitation nodes')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=500, help='max iteration step')
parser.add_argument('--n_embedding', dest='n_embedding', type=int, default=32, help='')

args = parser.parse_args()

def main(_):


	def construct_graph_network(net_source_file):
		a = pd.read_csv(net_source_file)
		a.columns=['target','source']
		edges = []
		for i in np.array(a):
			edges.append(i)
		G = nx.Graph()
		G.add_edges_from(edges)
		return G

	h = construct_graph_network('.\\radiation_fd_PPI.csv')
	adjacency_mat=nx.to_numpy_matrix(h)
	n_nodes=adjacency_mat.shape[0]
	config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	with tf.compat.v1.Session(config=config) as sess:
		model = VGAE(sess, n_nodes, args)
		model.train(args, adjacency_mat)
if __name__ == '__main__':
	tf.compat.v1.app.run()