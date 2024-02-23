import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from mcd_nn_priorization import OCC_training_module_and_gene_prediction_module as OCC
import networkx as nx
parser = argparse.ArgumentParser(description='')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=500, help='max iteration step')#500
args = parser.parse_args()
def main(_):
    def construct_graph_network(net_source_file):
        a = pd.read_csv(net_source_file)
        a.columns = ['target', 'source']
        edges = []
        for i in np.array(a):
            edges.append(i)
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    annotation_file = pd.read_csv('radiation_related_gene_annotation.csv', )#read annotation filr
    positive_index = list(annotation_file.loc[annotation_file['label'] == 1].index)#get the indexes of positive_examples
    G = construct_graph_network('radiation_fd_PPI.csv')#read fd_PPI file
    adjacency_mat = nx.to_numpy_matrix(G)
    n_nodes = adjacency_mat.shape[0]
    nodes_index = list(G.nodes())
    nodes_index= [nodes_index.index(i) for i in range(len(nodes_index))]
    embeddings_file = pd.read_csv('radiation_embedding.csv', header=None)#read embedding file from Feature_Representation Module generation file
    embeddings_file = embeddings_file.loc[nodes_index, :]
    embeddings_file.index = [i for i in range(embeddings_file.shape[0])]
    dff111 = embeddings_file
    positive_embeddings = dff111.loc[positive_index, :]#positive embeddings
    positive_embeddings.index = [i for i in range(positive_embeddings.shape[0])]
    X_train = positive_embeddings.iloc[0:299, :]#training positive examples
    X_test = positive_embeddings.iloc[299:590, :]#test positive examples
    unlabeled_index = list(annotation_file.loc[annotation_file['label'] == 0].index)#the indexes of unlabeled examples
    unlabeled_embeddings = embeddings_file.loc[unlabeled_index, :]#unlabeled embeddings
    unlabeled_embeddings.index = [i for i in range(unlabeled_embeddings.shape[0])]
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        model = OCC(sess, n_nodes, X_train, X_test, unlabeled_embeddings, args)
        model.train(args, X_train, X_test, unlabeled_embeddings)
if __name__ == '__main__':
    tf.compat.v1.app.run()
