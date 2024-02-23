import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import File_Reader
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
class Feature_representation_module(object):
    
        def __init__(self, sess, n_nodes, args):

            self.sess = sess
            self.result_dir = args.result_dir
            self.dataset_name = args.dataset_name
            self.n_nodes = n_nodes
            self.n_hidden = args.n_hidden
            self.n_embedding = args.n_embedding
            self.dropout = args.dropout
            self.learning_rate = args.learning_rate
            self.max_iteration = args.max_iteration                   
            self.shape = np.array([self.n_nodes, self.n_nodes])

            self.adjacency = tf.compat.v1.sparse_placeholder(tf.float32, shape=self.shape, name='adjacency')
            self.norm_adj_mat = tf.compat.v1.sparse_placeholder(tf.float32, shape=self.shape, name='norm_adj_mat')
            self.keep_prob = tf.compat.v1.placeholder(tf.float32)
            
            self.W_0_mu = None
            self.W_1_mu = None            
            self.W_0_sigma = None
            self.W_1_sigma = None
            
            self.mu_np = []
            self.sigma_np = []        
            
            self._build_Feature_representation_module()

            
        def _build_Feature_representation_module(self):

            z_encoded = self.encode()
            self.emb = z_encoded
            matrix_pred = self.decode()
            
            self.latent_loss = -(0.5/self.n_nodes)*tf.reduce_mean(tf.reduce_sum(1+2*tf.compat.v1.log(self.sigma)-tf.square(self.mu)-tf.square(self.sigma), 1))
            dense_adjacency = tf.reshape(tf.compat.v1.sparse_tensor_to_dense(self.adjacency, validate_indices=False), self.shape)
            w_1 = (self.n_nodes*self.n_nodes-tf.reduce_sum(dense_adjacency))/tf.reduce_sum(dense_adjacency)
            w_2 = self.n_nodes*self.n_nodes/(self.n_nodes*self.n_nodes-tf.reduce_sum(dense_adjacency))
            self.reconst_loss = w_2*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=dense_adjacency,\
                                                                                             logits=matrix_pred,\
                                                                                             pos_weight=w_1))
         
            self.loss = self.reconst_loss+self.latent_loss

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            self.train_step = self.optimizer.minimize(self.loss)

            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)
            

        def encode(self):

            self.W_0_mu = utils.unif_weight_init(shape=[self.n_nodes, self.n_hidden])
            self.b_0_mu = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden]))
            self.W_1_mu = utils.unif_weight_init(shape=[self.n_hidden, self.n_embedding])
            self.b_1_mu = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_embedding]))

            self.W_0_sigma = utils.unif_weight_init(shape=[self.n_nodes, self.n_hidden])
            self.b_0_sigma = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden]))
            self.W_1_sigma = utils.unif_weight_init(shape=[self.n_hidden, self.n_embedding])
            self.b_1_sigma = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_embedding]))

            hidden_0_mu_ = utils.gcn_layer_id(self.norm_adj_mat, self.W_0_mu, self.b_0_mu)
            if self.dropout:
                hidden_0_mu = tf.nn.dropout(hidden_0_mu_, self.keep_prob)
            else:
                hidden_0_mu = hidden_0_mu_
            self.mu = utils.gcn_layer1(self.norm_adj_mat, hidden_0_mu, self.W_1_mu, self.b_1_mu)
            
            hidden_0_sigma_ = utils.gcn_layer_id(self.norm_adj_mat, self.W_0_sigma, self.b_0_sigma)
            if self.dropout:
                hidden_0_sigma = tf.nn.dropout(hidden_0_sigma_, self.keep_prob)
            else:
                hidden_0_sigma = hidden_0_sigma_
            log_sigma = utils.gcn_layer1(self.norm_adj_mat, hidden_0_sigma, self.W_1_sigma, self.b_1_sigma)
            self.sigma = tf.exp(log_sigma)

            return utils.sample_gaussian(self.mu, self.sigma)
        

        def decode(self):

            z = utils.sample_gaussian(self.mu, self.sigma)
            matrix_pred = tf.matmul(z, z, transpose_a=False, transpose_b=True)
            return matrix_pred
      

        def train(self, args, adjacency):
            train_test_split = File_Reader.train_test_split(adjacency)
            train_adjacency = train_test_split[0]
            sp_adjacency = File_Reader.dense_to_sparse(train_adjacency)
            norm_adj_mat = File_Reader.normalize_adjacency(train_adjacency)
            feed_dict = {self.adjacency:sp_adjacency[0:2], self.norm_adj_mat:norm_adj_mat[0:2], self.keep_prob:args.keep_prob}
            for i in range(self.max_iteration):
                _, loss, latent_loss, reconst_loss, self.mu_np, self.sigma_np, embedding = self.sess.run([self.train_step,\
                                                                                           self.loss,\
                                                                                           self.latent_loss,\
                                                                                           self.reconst_loss,\
                                                                                           self.mu,\
                                                                                           self.sigma,\
                                                                                           self.emb],\

                                                                                           feed_dict=feed_dict)

                print("At step {0} \n Loss: {1} \n.".format(i, loss))


            np.savetxt('.\\radiation_embedding.csv', embedding, delimiter=',')#each  embedding of the gene



        

